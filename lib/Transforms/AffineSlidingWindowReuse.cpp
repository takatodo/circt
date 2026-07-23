//===- AffineSlidingWindowReuse.cpp - Reuse affine load windows -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AffineSlidingReuseSupport.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>
#include <optional>

namespace circt {
#define GEN_PASS_DEF_AFFINESLIDINGWINDOWREUSE
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::affine;
using circt::affine_reuse_detail::CheckedLinearExpr;
using circt::affine_reuse_detail::CheckedLinearExprExpander;
using circt::affine_reuse_detail::findModifyingOperation;
using circt::affine_reuse_detail::getExactPositiveTripCount;

namespace {

struct LoadAtOffset {
  AffineLoadOp load;
  int64_t offset;
};

struct SlidingWindow {
  Value source;
  SmallVector<LoadAtOffset> loads;
};

/// Return the constant difference between an affine load's rank-one access and
/// the loop induction variable. Checked composition recognizes equivalent
/// affine.apply chains without overflowing or unbounded recursion.
static std::optional<int64_t>
getConstantOffset(AffineLoadOp load, AffineForOp loop,
                  CheckedLinearExprExpander &accessExpander) {
  auto memrefType = dyn_cast<MemRefType>(load.getMemRef().getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      load.getAffineMap().getNumResults() != 1)
    return std::nullopt;

  std::optional<CheckedLinearExpr> access = accessExpander.expand(
      load.getAffineMap().getResult(0), load.getMapOperands(),
      load.getAffineMap().getNumDims());
  if (!access || access->inductionCoefficient != 1 ||
      !access->invariantCoefficients.empty())
    return std::nullopt;
  return access->constant;
}

/// Return true if an operation in the loop may modify the source. Unknown
/// effects intentionally block the transformation.
static bool mayModifySource(AffineForOp loop, Value source,
                            AliasAnalysis &aliasAnalysis) {
  return findModifyingOperation(loop, source, aliasAnalysis) != nullptr;
}

/// Return true only when every candidate trip-count expression is a signed
/// constant greater than one. A single iteration has no cross-iteration reuse.
/// Checking the expressions directly also avoids turning a negative symbolic
/// span into a large uint64_t trip count.
static bool hasProfitableConstantTripCount(AffineForOp loop) {
  std::optional<uint64_t> tripCount = getExactPositiveTripCount(loop);
  return tripCount && *tripCount > 1;
}

/// Find one duplicate-free contiguous window. Gaps between recognized offsets
/// split independent windows on the same source. Loads whose access is not IV
/// plus a constant are intentionally ignored and left untouched. Sources are
/// considered in their first-use order for deterministic rewriting.
static std::optional<SlidingWindow>
findSlidingWindow(AffineForOp loop, AliasAnalysis &aliasAnalysis) {
  if (loop.getStepAsInt() != 1 || loop.getLowerBoundMap().getNumResults() != 1)
    return std::nullopt;

  if (!hasProfitableConstantTripCount(loop))
    return std::nullopt;

  CheckedLinearExprExpander accessExpander(loop.getInductionVar());
  llvm::MapVector<Value, SmallVector<LoadAtOffset>> loadsBySource;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto load = dyn_cast<AffineLoadOp>(operation);
    if (!load)
      continue;
    std::optional<int64_t> offset =
        getConstantOffset(load, loop, accessExpander);
    if (!offset)
      continue;
    loadsBySource[load.getMemRef()].push_back({load, *offset});
  }

  for (auto &[source, loads] : loadsBySource) {
    if (loads.size() < 2 || !loop.isDefinedOutsideOfLoop(source))
      continue;

    llvm::sort(loads, [](const LoadAtOffset &lhs, const LoadAtOffset &rhs) {
      return lhs.offset < rhs.offset;
    });

    // A duplicate recognized offset makes the access-to-state correspondence
    // ambiguous, so conservatively reject every window on this source.
    bool hasDuplicate = false;
    for (size_t i = 1; i < loads.size(); ++i) {
      if (loads[i].offset == loads[i - 1].offset) {
        hasDuplicate = true;
        break;
      }
    }
    if (hasDuplicate || mayModifySource(loop, source, aliasAnalysis))
      continue;

    for (size_t windowBegin = 0; windowBegin < loads.size();) {
      size_t windowEnd = windowBegin + 1;
      while (windowEnd < loads.size() &&
             loads[windowEnd - 1].offset !=
                 std::numeric_limits<int64_t>::max() &&
             loads[windowEnd].offset == loads[windowEnd - 1].offset + 1)
        ++windowEnd;

      if (windowEnd - windowBegin >= 2) {
        SmallVector<LoadAtOffset> windowLoads(loads.begin() + windowBegin,
                                              loads.begin() + windowEnd);
        return SlidingWindow{source, std::move(windowLoads)};
      }
      windowBegin = windowEnd;
    }
  }
  return std::nullopt;
}

/// Materialize the first window before the loop, append its scalar state to
/// the affine.for iter_args, and rotate that state on every iteration.
static FailureOr<AffineForOp> rewriteSlidingWindow(IRRewriter &rewriter,
                                                   AffineForOp loop,
                                                   SlidingWindow window) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  SmallVector<Value> preloads;
  preloads.reserve(window.loads.size() - 1);
  AffineMap lowerBoundMap = loop.getLowerBoundMap();
  ValueRange lowerBoundOperands = loop.getLowerBoundOperands();
  DictionaryAttr loopAttrs = loop->getDiscardableAttrDictionary();
  for (LoadAtOffset &access : llvm::drop_end(window.loads)) {
    AffineExpr index = lowerBoundMap.getResult(0) + access.offset;
    AffineMap preloadMap = AffineMap::get(lowerBoundMap.getNumDims(),
                                          lowerBoundMap.getNumSymbols(), index);
    auto preload =
        AffineLoadOp::create(rewriter, access.load.getLoc(), window.source,
                             preloadMap, lowerBoundOperands);
    preload->setDiscardableAttrs(access.load->getDiscardableAttrDictionary());
    preloads.push_back(preload);
  }

  AffineLoadOp nextLoad = window.loads.back().load;
  auto maybeNewLoop = loop.replaceWithAdditionalYields(
      rewriter, preloads, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument> newWindowArgs) {
        SmallVector<Value> nextWindow;
        llvm::append_range(nextWindow, newWindowArgs.drop_front());
        nextWindow.push_back(nextLoad.getResult());
        return nextWindow;
      });
  if (failed(maybeNewLoop)) {
    for (Value preload : llvm::reverse(preloads))
      rewriter.eraseOp(preload.getDefiningOp());
    return failure();
  }

  auto newLoop = cast<AffineForOp>(maybeNewLoop->getOperation());
  newLoop->setDiscardableAttrs(loopAttrs);
  ValueRange windowArgs =
      newLoop.getRegionIterArgs().take_back(preloads.size());
  for (auto [access, windowArg] :
       llvm::zip(llvm::drop_end(window.loads), windowArgs))
    rewriter.replaceOp(access.load, windowArg);

  return newLoop;
}

struct AffineSlidingWindowReusePass
    : public circt::impl::AffineSlidingWindowReuseBase<
          AffineSlidingWindowReusePass> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    IRRewriter rewriter(&getContext());

    // Rebuild one loop at a time. Restarting also lets a loop with independent
    // windows on multiple sources be handled without keeping stale op handles.
    while (true) {
      bool changed = false;
      SmallVector<AffineForOp> loops;
      getOperation().walk<WalkOrder::PostOrder>(
          [&](AffineForOp loop) { loops.push_back(loop); });
      for (AffineForOp loop : loops) {
        std::optional<SlidingWindow> window =
            findSlidingWindow(loop, aliasAnalysis);
        if (!window)
          continue;
        if (failed(rewriteSlidingWindow(rewriter, loop, std::move(*window)))) {
          signalPassFailure();
          return;
        }
        changed = true;
        break;
      }
      if (!changed)
        break;
    }
  }
};

} // namespace
