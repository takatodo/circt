//===- AffineSlidingReductionReuse.cpp - Carry sliding folds -*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Match affine sliding-window reductions and carry the partial fold of each
// leading column through affine.for iter_args.
//
//===----------------------------------------------------------------------===//

#include "AffineSlidingReuseAnalysis.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <limits>
#include <optional>

namespace circt {
#define GEN_PASS_DEF_AFFINESLIDINGREDUCTIONREUSE
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::affine;
using circt::affine_reuse_detail::CheckedLinearExpr;
using circt::affine_reuse_detail::CheckedLinearExprExpander;
using circt::affine_reuse_detail::getExactPositiveTripCount;
using circt::affine_reuse_detail::isFirstIterationLoadCompositionRepresentable;
using circt::affine_reuse_detail::mayModifySource;

namespace {

enum class ReductionKind { IntegerAdd, FloatAdd, IntegerMul, SignedIntegerMax };

/// One normalized affine-map result. The sliding result has its varying
/// constant removed so otherwise-identical members in adjacent columns share
/// a key.
struct NormalizedSubscript {
  int64_t constant;
  SmallVector<std::pair<Value, int64_t>, 4> invariantCoefficients;

  bool operator==(const NormalizedSubscript &other) const {
    return constant == other.constant &&
           invariantCoefficients == other.invariantCoefficients;
  }
};

/// The complete access identity except for the sliding column offset.
struct AccessKey {
  SmallVector<NormalizedSubscript, 4> subscripts;
  unsigned slidingPosition = 0;

  bool operator==(const AccessKey &other) const {
    return slidingPosition == other.slidingPosition &&
           subscripts == other.subscripts;
  }
};

struct WindowLoad {
  AffineLoadOp load;
  int64_t slidingOffset;
  AccessKey key;
};

/// A fully checked plan consumed by the register-carry rewrite.
struct ReductionWindow {
  Operation *root;
  Value source;
  SmallVector<Operation *> internalOps;
  SmallVector<SmallVector<AffineLoadOp>> columns;
  ReductionKind reductionKind;
  arith::FastMathFlags fastMathFlags;
};

static std::optional<ReductionKind> getReductionKind(Operation *operation) {
  if (isa<arith::AddIOp>(operation))
    return ReductionKind::IntegerAdd;
  if (isa<arith::AddFOp>(operation))
    return ReductionKind::FloatAdd;
  if (isa<arith::MulIOp>(operation))
    return ReductionKind::IntegerMul;
  if (isa<arith::MaxSIOp>(operation))
    return ReductionKind::SignedIntegerMax;
  return std::nullopt;
}

/// Check that regrouping cannot change poison or floating-point semantics.
/// The intersection of floating-point flags is attached to every new node.
static bool updateReductionSemantics(Operation *operation,
                                     ReductionKind reductionKind,
                                     arith::FastMathFlags &fastMathFlags) {
  switch (reductionKind) {
  case ReductionKind::IntegerAdd:
    return cast<arith::AddIOp>(operation).getOverflowFlags() ==
           arith::IntegerOverflowFlags::none;
  case ReductionKind::FloatAdd: {
    auto add = cast<arith::AddFOp>(operation);
    if (add.getRoundingmodeAttr() ||
        !arith::bitEnumContainsAll(add.getFastmath(),
                                   arith::FastMathFlags::reassoc))
      return false;
    fastMathFlags = fastMathFlags & add.getFastmath();
    return true;
  }
  case ReductionKind::IntegerMul:
    return cast<arith::MulIOp>(operation).getOverflowFlags() ==
           arith::IntegerOverflowFlags::none;
  case ReductionKind::SignedIntegerMax:
    return true;
  }
  llvm_unreachable("unknown reduction kind");
}

/// Match one maximal same-operation tree and group its direct affine loads by
/// their offset along the loop induction variable.
static std::optional<ReductionWindow>
matchReductionTree(Operation *root, AffineForOp loop,
                   CheckedLinearExprExpander &accessExpander) {
  std::optional<ReductionKind> reductionKind = getReductionKind(root);
  if (!reductionKind || root->use_empty())
    return std::nullopt;

  arith::FastMathFlags fastMathFlags = arith::FastMathFlags::fast;
  SmallVector<Operation *> internalOps;
  SmallVector<AffineLoadOp> leaves;
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!worklist.empty()) {
    Operation *node = worklist.pop_back_val();
    if (!seen.insert(node).second ||
        !updateReductionSemantics(node, *reductionKind, fastMathFlags))
      return std::nullopt;

    for (Value operand : node->getOperands()) {
      Operation *defining = operand.getDefiningOp();
      bool isInternal = defining && defining->getName() == root->getName() &&
                        defining->getBlock() == root->getBlock();
      if (isInternal) {
        // A shared internal value cannot be erased or regrouped as one tree.
        if (!operand.hasOneUse())
          return std::nullopt;
        internalOps.push_back(defining);
        worklist.push_back(defining);
        continue;
      }

      AffineLoadOp load = operand.getDefiningOp<AffineLoadOp>();
      if (!load || load->getBlock() != loop.getBody())
        return std::nullopt;
      // Only the induction variable may be remapped when the load is cloned
      // into the prologue. Any other in-loop operand would escape dominance.
      for (Value mapOperand : load.getMapOperands())
        if (mapOperand != loop.getInductionVar() &&
            !loop.isDefinedOutsideOfLoop(mapOperand))
          return std::nullopt;
      leaves.push_back(load);
    }
  }
  if (leaves.size() < 2)
    return std::nullopt;

  Value source = leaves.front().getMemRef();
  if (!isa<MemRefType>(source.getType()) ||
      !loop.isDefinedOutsideOfLoop(source) ||
      llvm::any_of(leaves, [&](AffineLoadOp load) {
        return load.getMemRef() != source;
      }))
    return std::nullopt;

  std::optional<unsigned> commonSlidingPosition;
  SmallVector<WindowLoad, 0> windowLoads;
  for (AffineLoadOp load : leaves) {
    AffineMap map = load.getAffineMap();
    AccessKey key;
    std::optional<int64_t> slidingOffset;
    for (unsigned position = 0; position < map.getNumResults(); ++position) {
      std::optional<CheckedLinearExpr> expression = accessExpander.expand(
          map.getResult(position), load.getMapOperands(), map.getNumDims());
      if (!expression ||
          llvm::any_of(expression->invariantCoefficients,
                       [&](const auto &term) {
                         return !loop.isDefinedOutsideOfLoop(term.first);
                       }))
        return std::nullopt;

      NormalizedSubscript normalized{expression->constant,
                                     expression->invariantCoefficients};
      if (expression->inductionCoefficient == 1) {
        if (slidingOffset)
          return std::nullopt;
        slidingOffset = expression->constant;
        normalized.constant = 0;
        if (!commonSlidingPosition)
          commonSlidingPosition = position;
        else if (*commonSlidingPosition != position)
          return std::nullopt;
        key.slidingPosition = position;
      } else if (expression->inductionCoefficient != 0) {
        return std::nullopt;
      }
      key.subscripts.push_back(std::move(normalized));
    }
    if (!slidingOffset)
      return std::nullopt;
    windowLoads.push_back({load, *slidingOffset, std::move(key)});
  }

  llvm::sort(windowLoads, [](const WindowLoad &lhs, const WindowLoad &rhs) {
    return lhs.slidingOffset < rhs.slidingOffset;
  });
  llvm::MapVector<int64_t, SmallVector<WindowLoad, 0>> byOffset;
  for (WindowLoad &windowLoad : windowLoads)
    byOffset[windowLoad.slidingOffset].push_back(std::move(windowLoad));
  if (byOffset.size() < 2)
    return std::nullopt;

  int64_t previousOffset = byOffset.begin()->first;
  for (auto &[offset, members] : llvm::drop_begin(byOffset)) {
    if (previousOffset == std::numeric_limits<int64_t>::max() ||
        offset != previousOffset + 1)
      return std::nullopt;
    previousOffset = offset;
  }

  // The first column fixes deterministic row order. Every later column must
  // contain exactly one member with each normalized access key.
  SmallVector<WindowLoad, 0> &firstColumn = byOffset.begin()->second;
  llvm::sort(firstColumn, [](const WindowLoad &lhs, const WindowLoad &rhs) {
    return lhs.load->isBeforeInBlock(rhs.load);
  });

  ReductionWindow window{root, source,         std::move(internalOps),
                         {},   *reductionKind, fastMathFlags};
  for (auto &[offset, members] : byOffset) {
    if (members.size() != firstColumn.size())
      return std::nullopt;
    SmallVector<bool> consumed(members.size(), false);
    SmallVector<AffineLoadOp> column;
    for (const WindowLoad &reference : firstColumn) {
      std::optional<size_t> match;
      for (size_t i = 0; i < members.size(); ++i)
        if (!consumed[i] && members[i].key == reference.key) {
          match = i;
          break;
        }
      if (!match)
        return std::nullopt;
      consumed[*match] = true;
      column.push_back(members[*match].load);
    }
    window.columns.push_back(std::move(column));
  }

  // Carrying a value does not save its load when that load also has another
  // user. Leave cost modeling for partially reusable columns to a later pass.
  for (ArrayRef<AffineLoadOp> column : ArrayRef(window.columns).drop_back()) {
    if (llvm::any_of(column,
                     [](AffineLoadOp load) { return !load->hasOneUse(); }))
      return std::nullopt;
    if (llvm::any_of(column, [&](AffineLoadOp load) {
          return !isFirstIterationLoadCompositionRepresentable(load, loop,
                                                               accessExpander);
        }))
      return std::nullopt;
  }
  return window;
}

/// Return true for a top-level tree or a subtree with an additional,
/// non-reduction use. Purely internal nodes are visited from such a root.
static bool isReductionTreeRoot(Operation *operation) {
  bool hasSameKindUser = false;
  for (Operation *user : operation->getUsers()) {
    if (user->getName() == operation->getName() &&
        user->getBlock() == operation->getBlock()) {
      hasSameKindUser = true;
      continue;
    }
    return true;
  }
  return !hasSameKindUser;
}

/// Collect the root followed by its same-kind subtrees. This lets matching
/// fall back to a load-only subtree when the outer operation also consumes a
/// seed or loop-carried accumulator.
static void collectReductionSubtrees(Operation *root,
                                     SmallVectorImpl<Operation *> &candidates) {
  SmallVector<Operation *> worklist{root};
  llvm::SmallPtrSet<Operation *, 16> seen;
  while (!worklist.empty()) {
    Operation *node = worklist.pop_back_val();
    if (!seen.insert(node).second)
      continue;
    candidates.push_back(node);
    for (Value operand : node->getOperands()) {
      Operation *defining = operand.getDefiningOp();
      if (defining && defining->getName() == root->getName() &&
          defining->getBlock() == root->getBlock())
        worklist.push_back(defining);
    }
  }
}

/// Visit roots in block order and their same-kind subtrees in deterministic
/// root-first order, returning the first legal and potentially profitable
/// window.
static std::optional<ReductionWindow>
findReductionWindow(AffineForOp loop, AliasAnalysis &aliasAnalysis) {
  std::optional<uint64_t> tripCount = getExactPositiveTripCount(loop);
  if (!tripCount || *tripCount <= 1)
    return std::nullopt;

  CheckedLinearExprExpander accessExpander(loop.getInductionVar());
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (!getReductionKind(&operation) || operation.use_empty() ||
        !isReductionTreeRoot(&operation))
      continue;
    SmallVector<Operation *> candidates;
    collectReductionSubtrees(&operation, candidates);
    for (Operation *candidate : candidates) {
      std::optional<ReductionWindow> window =
          matchReductionTree(candidate, loop, accessExpander);
      if (!window || mayModifySource(loop, window->source, aliasAnalysis))
        continue;
      return window;
    }
  }
  return std::nullopt;
}

static Value buildFold(OpBuilder &builder, Location location, ValueRange values,
                       ReductionKind reductionKind,
                       arith::FastMathFlags fastMathFlags,
                       SmallVectorImpl<Operation *> &createdOps) {
  Value result = values.front();
  for (Value value : values.drop_front()) {
    switch (reductionKind) {
    case ReductionKind::IntegerAdd:
      result = arith::AddIOp::create(builder, location, result, value);
      break;
    case ReductionKind::FloatAdd:
      result = arith::AddFOp::create(
          builder, location, result, value,
          arith::FastMathFlagsAttr::get(builder.getContext(), fastMathFlags));
      break;
    case ReductionKind::IntegerMul:
      result = arith::MulIOp::create(builder, location, result, value);
      break;
    case ReductionKind::SignedIntegerMax:
      result = arith::MaxSIOp::create(builder, location, result, value);
      break;
    }
    createdOps.push_back(result.getDefiningOp());
  }
  return result;
}

static Value
cloneLoadAtFirstIteration(RewriterBase &rewriter, AffineLoadOp load,
                          BlockArgument inductionVariable, Value lowerBound,
                          SmallVectorImpl<Operation *> &createdOps) {
  IRMapping mapping;
  mapping.map(inductionVariable, lowerBound);
  Operation *clone = rewriter.clone(*load, mapping);
  createdOps.push_back(clone);
  return clone->getResult(0);
}

/// Materialize leading column folds in the prologue and rotate them through
/// new loop-carried values.
static LogicalResult rewriteReductionWindow(RewriterBase &rewriter,
                                            AffineForOp loop,
                                            ReductionWindow window) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location location = window.root->getLoc();
  size_t columnCount = window.columns.size();
  SmallVector<Operation *> createdOps;
  DictionaryAttr loopAttrs = loop->getDiscardableAttrDictionary();

  rewriter.setInsertionPoint(loop);
  Value lowerBound;
  if (auto constant =
          dyn_cast<AffineConstantExpr>(loop.getLowerBoundMap().getResult(0)))
    lowerBound =
        arith::ConstantIndexOp::create(rewriter, location, constant.getValue());
  else
    lowerBound =
        AffineApplyOp::create(rewriter, location, loop.getLowerBoundMap(),
                              loop.getLowerBoundOperands());
  createdOps.push_back(lowerBound.getDefiningOp());

  SmallVector<Value> initialFolds;
  for (ArrayRef<AffineLoadOp> column : ArrayRef(window.columns).drop_back()) {
    SmallVector<Value> values;
    for (AffineLoadOp load : column)
      values.push_back(cloneLoadAtFirstIteration(
          rewriter, load, loop.getInductionVar(), lowerBound, createdOps));
    initialFolds.push_back(buildFold(rewriter, location, values,
                                     window.reductionKind, window.fastMathFlags,
                                     createdOps));
  }

  rewriter.setInsertionPoint(window.root);
  SmallVector<Value> newestValues(
      llvm::map_range(window.columns.back(),
                      [](AffineLoadOp load) { return load.getResult(); }));
  Value newestFold =
      buildFold(rewriter, location, newestValues, window.reductionKind,
                window.fastMathFlags, createdOps);

  auto maybeNewLoop = loop.replaceWithAdditionalYields(
      rewriter, initialFolds, /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument> newArguments) {
        SmallVector<Value> nextState;
        llvm::append_range(nextState, newArguments.drop_front());
        nextState.push_back(newestFold);
        return nextState;
      });
  if (failed(maybeNewLoop)) {
    for (Operation *created : llvm::reverse(createdOps))
      rewriter.eraseOp(created);
    return failure();
  }
  auto newLoop = cast<AffineForOp>(maybeNewLoop->getOperation());
  newLoop->setDiscardableAttrs(loopAttrs);

  rewriter.setInsertionPoint(window.root);
  ValueRange carried = newLoop.getRegionIterArgs().take_back(columnCount - 1);
  SmallVector<Value> allColumns(carried.begin(), carried.end());
  allColumns.push_back(newestFold);
  Value combined =
      buildFold(rewriter, location, allColumns, window.reductionKind,
                window.fastMathFlags, createdOps);
  rewriter.replaceAllUsesWith(window.root->getResult(0), combined);

  // Erase the old fold and its leading loads only after they become dead.
  SmallVector<Operation *> deadCandidates{window.root};
  llvm::append_range(deadCandidates, window.internalOps);
  for (ArrayRef<AffineLoadOp> column : ArrayRef(window.columns).drop_back())
    for (AffineLoadOp load : column)
      deadCandidates.push_back(load);
  bool erased = true;
  while (erased) {
    erased = false;
    for (Operation *&candidate : deadCandidates)
      if (candidate && candidate->use_empty()) {
        rewriter.eraseOp(candidate);
        candidate = nullptr;
        erased = true;
      }
  }
  return success();
}

static FailureOr<bool> rewriteOneReduction(RewriterBase &rewriter,
                                           AffineForOp loop,
                                           AliasAnalysis &aliasAnalysis) {
  std::optional<ReductionWindow> window =
      findReductionWindow(loop, aliasAnalysis);
  if (!window)
    return false;
  if (failed(rewriteReductionWindow(rewriter, loop, std::move(*window))))
    return failure();
  return true;
}

struct AffineSlidingReductionReusePass
    : public circt::impl::AffineSlidingReductionReuseBase<
          AffineSlidingReductionReusePass> {
  void runOnOperation() override {
    AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
    IRRewriter rewriter(&getContext());

    // Rebuild one loop at a time. Restarting avoids stale handles and also
    // handles independent windows in the same loop.
    while (true) {
      bool changed = false;
      SmallVector<AffineForOp> loops;
      getOperation().walk<WalkOrder::PostOrder>(
          [&](AffineForOp loop) { loops.push_back(loop); });
      for (AffineForOp loop : loops) {
        FailureOr<bool> rewritten =
            rewriteOneReduction(rewriter, loop, aliasAnalysis);
        if (failed(rewritten)) {
          signalPassFailure();
          return;
        }
        if (!*rewritten)
          continue;
        changed = true;
        break;
      }
      if (!changed)
        break;
    }
  }
};

} // namespace
