//===- AffineSlidingWindowReuse.cpp - Reuse affine load windows -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <optional>

namespace circt {
#define GEN_PASS_DEF_AFFINESLIDINGWINDOWREUSE
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::affine;

namespace {

struct LoadAtOffset {
  AffineLoadOp load;
  int64_t offset;
};

struct SlidingWindow {
  Value source;
  SmallVector<LoadAtOffset> loads;
};

struct CheckedLinearExpr {
  int64_t inductionCoefficient;
  int64_t constant;
};

using CheckedLinearValueCache =
    llvm::DenseMap<Value, std::optional<CheckedLinearExpr>>;

/// Keep malformed or adversarial affine.apply chains from exhausting the C++
/// stack. Exceeding this proof limit only makes the matcher conservative.
static constexpr unsigned maxAffineExpansionDepth = 256;

static std::optional<CheckedLinearExpr>
expandCheckedLinearExpr(AffineExpr expression, ValueRange operands,
                        unsigned numDims, Value inductionVariable,
                        CheckedLinearValueCache &valueCache, unsigned depth);

/// Expand a map operand without letting affine composition overflow while
/// folding constant SSA values or affine.apply chains.
static std::optional<CheckedLinearExpr>
expandCheckedLinearValue(Value value, Value inductionVariable,
                         CheckedLinearValueCache &valueCache, unsigned depth) {
  auto cached = valueCache.find(value);
  if (cached != valueCache.end())
    return cached->second;
  if (depth > maxAffineExpansionDepth)
    return std::nullopt;

  std::optional<CheckedLinearExpr> expansion;
  if (value == inductionVariable) {
    expansion = CheckedLinearExpr{1, 0};
  } else if (IntegerAttr constant; matchPattern(value, m_Constant(&constant))) {
    const llvm::APInt &integer = constant.getValue();
    if (integer.isSignedIntN(64))
      expansion = CheckedLinearExpr{0, integer.getSExtValue()};
  } else if (auto apply = value.getDefiningOp<AffineApplyOp>()) {
    expansion = expandCheckedLinearExpr(
        apply.getAffineMap().getResult(0), apply.getMapOperands(),
        apply.getAffineMap().getNumDims(), inductionVariable, valueCache,
        depth + 1);
  }
  valueCache.insert({value, expansion});
  return expansion;
}

static std::optional<CheckedLinearExpr>
expandCheckedLinearExpr(AffineExpr expression, ValueRange operands,
                        unsigned numDims, Value inductionVariable,
                        CheckedLinearValueCache &valueCache, unsigned depth) {
  if (depth > maxAffineExpansionDepth)
    return std::nullopt;
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return CheckedLinearExpr{0, constant.getValue()};
  if (auto dim = dyn_cast<AffineDimExpr>(expression))
    return expandCheckedLinearValue(operands[dim.getPosition()],
                                    inductionVariable, valueCache, depth + 1);
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
    return expandCheckedLinearValue(operands[numDims + symbol.getPosition()],
                                    inductionVariable, valueCache, depth + 1);

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  std::optional<CheckedLinearExpr> lhs =
      expandCheckedLinearExpr(binary.getLHS(), operands, numDims,
                              inductionVariable, valueCache, depth + 1);
  std::optional<CheckedLinearExpr> rhs =
      expandCheckedLinearExpr(binary.getRHS(), operands, numDims,
                              inductionVariable, valueCache, depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;

  if (expression.getKind() == AffineExprKind::Add) {
    CheckedLinearExpr result;
    if (llvm::AddOverflow(lhs->inductionCoefficient, rhs->inductionCoefficient,
                          result.inductionCoefficient) ||
        llvm::AddOverflow(lhs->constant, rhs->constant, result.constant))
      return std::nullopt;
    return result;
  }

  if (expression.getKind() != AffineExprKind::Mul ||
      (lhs->inductionCoefficient != 0 && rhs->inductionCoefficient != 0))
    return std::nullopt;

  CheckedLinearExpr linear = lhs->inductionCoefficient == 0 ? *rhs : *lhs;
  int64_t factor =
      lhs->inductionCoefficient == 0 ? lhs->constant : rhs->constant;
  CheckedLinearExpr result;
  if (llvm::MulOverflow(linear.inductionCoefficient, factor,
                        result.inductionCoefficient) ||
      llvm::MulOverflow(linear.constant, factor, result.constant))
    return std::nullopt;
  return result;
}

/// Return the constant difference between a rank-one access and the induction
/// variable. Checked expansion recognizes affine.apply chains but rejects any
/// intermediate coefficient or constant arithmetic that overflows int64_t.
static std::optional<int64_t>
getConstantOffset(AffineLoadOp load, AffineForOp loop,
                  CheckedLinearValueCache &cache) {
  auto memrefType = dyn_cast<MemRefType>(load.getMemRef().getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      load.getAffineMap().getNumResults() != 1)
    return std::nullopt;

  std::optional<CheckedLinearExpr> access = expandCheckedLinearExpr(
      load.getAffineMap().getResult(0), load.getMapOperands(),
      load.getAffineMap().getNumDims(), loop.getInductionVar(), cache,
      /*depth=*/0);
  if (!access || access->inductionCoefficient != 1)
    return std::nullopt;
  return access->constant;
}

/// Return true if an operation in the loop may modify the source. Unknown
/// effects intentionally block the transformation.
static bool mayModifySource(AffineForOp loop, Value source,
                            AliasAnalysis &aliasAnalysis) {
  WalkResult result =
      loop.getBody()->walk<WalkOrder::PostOrder>([&](Operation *operation) {
        // Operations with recursive memory effects derive their effects from
        // nested operations when they do not also implement a direct effect
        // interface. The post-order walk has already checked those nested
        // operations, while querying LocalAliasAnalysis for the container would
        // conservatively return ModRef and reject even an empty nested loop.
        if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
            !isa<MemoryEffectOpInterface>(operation))
          return WalkResult::advance();
        if (aliasAnalysis.getModRef(operation, source).isMod())
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  return result.wasInterrupted();
}

/// An exact linear expression over SSA leaf values. APInt widths grow with the
/// expression so composing affine.apply chains cannot wrap at int64_t.
struct ExactLinearExpr {
  llvm::APInt constant = llvm::APInt(1, 0);
  llvm::MapVector<Value, llvm::APInt> coefficients;
};

using ExactLinearValueCache =
    llvm::DenseMap<Value, std::optional<ExactLinearExpr>>;

static llvm::APInt normalizeSigned(llvm::APInt value) {
  return value.sextOrTrunc(value.getSignificantBits());
}

static std::optional<llvm::APInt> addSignedExact(const llvm::APInt &lhs,
                                                 const llvm::APInt &rhs) {
  uint64_t width = static_cast<uint64_t>(std::max(lhs.getSignificantBits(),
                                                  rhs.getSignificantBits())) +
                   1;
  if (width > std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return normalizeSigned(lhs.sextOrTrunc(width) + rhs.sextOrTrunc(width));
}

static std::optional<llvm::APInt> subtractSignedExact(const llvm::APInt &lhs,
                                                      const llvm::APInt &rhs) {
  uint64_t width = static_cast<uint64_t>(std::max(lhs.getSignificantBits(),
                                                  rhs.getSignificantBits())) +
                   1;
  if (width > std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return normalizeSigned(lhs.sextOrTrunc(width) - rhs.sextOrTrunc(width));
}

static std::optional<llvm::APInt> multiplySignedExact(const llvm::APInt &lhs,
                                                      const llvm::APInt &rhs) {
  if (lhs.isZero() || rhs.isZero())
    return llvm::APInt(1, 0);
  uint64_t width = static_cast<uint64_t>(lhs.getSignificantBits()) +
                   rhs.getSignificantBits();
  if (width > std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return normalizeSigned(lhs.sextOrTrunc(width) * rhs.sextOrTrunc(width));
}

static std::optional<ExactLinearExpr>
addExactLinearExpr(const ExactLinearExpr &lhs, const ExactLinearExpr &rhs) {
  ExactLinearExpr result = lhs;
  std::optional<llvm::APInt> constant =
      addSignedExact(lhs.constant, rhs.constant);
  if (!constant)
    return std::nullopt;
  result.constant = std::move(*constant);

  for (const auto &[value, coefficient] : rhs.coefficients) {
    auto existing = result.coefficients.find(value);
    if (existing == result.coefficients.end()) {
      result.coefficients.insert({value, coefficient});
      continue;
    }
    std::optional<llvm::APInt> sum =
        addSignedExact(existing->second, coefficient);
    if (!sum)
      return std::nullopt;
    if (sum->isZero())
      result.coefficients.erase(value);
    else
      existing->second = std::move(*sum);
  }
  return result;
}

static std::optional<ExactLinearExpr>
scaleExactLinearExpr(const ExactLinearExpr &expression,
                     const llvm::APInt &factor) {
  ExactLinearExpr result;
  std::optional<llvm::APInt> constant =
      multiplySignedExact(expression.constant, factor);
  if (!constant)
    return std::nullopt;
  result.constant = std::move(*constant);
  for (const auto &[value, coefficient] : expression.coefficients) {
    std::optional<llvm::APInt> product =
        multiplySignedExact(coefficient, factor);
    if (!product)
      return std::nullopt;
    if (!product->isZero())
      result.coefficients.insert({value, std::move(*product)});
  }
  return result;
}

static std::optional<ExactLinearExpr>
expandExactLinearExpr(AffineExpr expression, ValueRange operands,
                      unsigned numDims, ExactLinearValueCache &valueCache,
                      unsigned depth);

static std::optional<ExactLinearExpr>
expandExactLinearValue(Value value, ExactLinearValueCache &valueCache,
                       unsigned depth) {
  auto cached = valueCache.find(value);
  if (cached != valueCache.end())
    return cached->second;
  if (depth > maxAffineExpansionDepth)
    return std::nullopt;

  std::optional<ExactLinearExpr> expansion;
  IntegerAttr constant;
  if (matchPattern(value, m_Constant(&constant))) {
    ExactLinearExpr result;
    result.constant = normalizeSigned(constant.getValue());
    expansion = std::move(result);
  } else if (auto apply = value.getDefiningOp<AffineApplyOp>()) {
    expansion = expandExactLinearExpr(
        apply.getAffineMap().getResult(0), apply.getMapOperands(),
        apply.getAffineMap().getNumDims(), valueCache, depth + 1);
  } else {
    ExactLinearExpr result;
    result.coefficients.insert({value, llvm::APInt(2, 1)});
    expansion = std::move(result);
  }
  valueCache.insert({value, expansion});
  return expansion;
}

/// Expand affine addition and multiplication by an exact constant. Other
/// semi-affine operations are conservatively unsupported.
static std::optional<ExactLinearExpr>
expandExactLinearExpr(AffineExpr expression, ValueRange operands,
                      unsigned numDims, ExactLinearValueCache &valueCache,
                      unsigned depth) {
  if (depth > maxAffineExpansionDepth)
    return std::nullopt;
  if (auto constant = dyn_cast<AffineConstantExpr>(expression)) {
    ExactLinearExpr result;
    result.constant = normalizeSigned(
        llvm::APInt(64, static_cast<uint64_t>(constant.getValue())));
    return result;
  }
  if (auto dim = dyn_cast<AffineDimExpr>(expression))
    return expandExactLinearValue(operands[dim.getPosition()], valueCache,
                                  depth + 1);
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
    return expandExactLinearValue(operands[numDims + symbol.getPosition()],
                                  valueCache, depth + 1);

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  std::optional<ExactLinearExpr> lhs = expandExactLinearExpr(
      binary.getLHS(), operands, numDims, valueCache, depth + 1);
  std::optional<ExactLinearExpr> rhs = expandExactLinearExpr(
      binary.getRHS(), operands, numDims, valueCache, depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;

  if (expression.getKind() == AffineExprKind::Add)
    return addExactLinearExpr(*lhs, *rhs);
  if (expression.getKind() != AffineExprKind::Mul)
    return std::nullopt;

  bool lhsIsConstant = lhs->coefficients.empty();
  bool rhsIsConstant = rhs->coefficients.empty();
  if (!lhsIsConstant && !rhsIsConstant)
    return std::nullopt;
  return lhsIsConstant ? scaleExactLinearExpr(*rhs, lhs->constant)
                       : scaleExactLinearExpr(*lhs, rhs->constant);
}

static bool haveSameCoefficients(const ExactLinearExpr &lhs,
                                 const ExactLinearExpr &rhs) {
  if (lhs.coefficients.size() != rhs.coefficients.size())
    return false;
  for (const auto &[value, lhsCoefficient] : lhs.coefficients) {
    auto rhsCoefficient = rhs.coefficients.find(value);
    if (rhsCoefficient == rhs.coefficients.end())
      return false;
    unsigned width = std::max(lhsCoefficient.getBitWidth(),
                              rhsCoefficient->second.getBitWidth());
    if (lhsCoefficient.sextOrTrunc(width) !=
        rhsCoefficient->second.sextOrTrunc(width))
      return false;
  }
  return true;
}

/// Return the exact positive trip count of a unit-step loop when each upper
/// bound is the lower bound plus a positive constant offset. The effective
/// affine upper bound is the minimum of its results. Bounds and affine.apply
/// chains are expanded with arbitrary-width signed arithmetic before their
/// symbolic coefficients are compared. Empty, nonlinear, and more complicated
/// bounds are conservatively unsupported.
static std::optional<uint64_t> getConstantTripCount(AffineForOp loop) {
  AffineMap lowerMap = loop.getLowerBoundMap();
  if (lowerMap.getNumResults() != 1)
    return std::nullopt;
  ExactLinearValueCache valueCache;
  std::optional<ExactLinearExpr> lower =
      expandExactLinearExpr(lowerMap.getResult(0), loop.getLowerBoundOperands(),
                            lowerMap.getNumDims(), valueCache, /*depth=*/0);
  if (!lower)
    return std::nullopt;

  AffineMap upperMap = loop.getUpperBoundMap();
  std::optional<uint64_t> minimumSpan;
  for (AffineExpr expression : upperMap.getResults()) {
    std::optional<ExactLinearExpr> upper =
        expandExactLinearExpr(expression, loop.getUpperBoundOperands(),
                              upperMap.getNumDims(), valueCache, /*depth=*/0);
    if (!upper || !haveSameCoefficients(*lower, *upper))
      return std::nullopt;

    std::optional<llvm::APInt> span =
        subtractSignedExact(upper->constant, lower->constant);
    if (!span || span->isNegative() || span->isZero() ||
        span->getActiveBits() > 64)
      return std::nullopt;
    uint64_t spanValue = span->getZExtValue();
    if (!minimumSpan || spanValue < *minimumSpan)
      minimumSpan = spanValue;
  }
  return minimumSpan;
}

/// Find one duplicate-free contiguous window. Gaps between recognized offsets
/// split independent windows on the same source. Loads whose access is not IV
/// plus a constant are intentionally ignored and left untouched. Sources are
/// considered in their first-use order for deterministic rewriting.
static std::optional<SlidingWindow>
findSlidingWindow(AffineForOp loop, AliasAnalysis &aliasAnalysis) {
  if (loop.getStepAsInt() != 1 || loop.getLowerBoundMap().getNumResults() != 1)
    return std::nullopt;

  std::optional<uint64_t> tripCount = getConstantTripCount(loop);
  if (!tripCount || *tripCount <= 1)
    return std::nullopt;

  llvm::MapVector<Value, SmallVector<LoadAtOffset>> loadsBySource;
  CheckedLinearValueCache accessValueCache;
  for (Operation &operation : loop.getBody()->without_terminator()) {
    auto load = dyn_cast<AffineLoadOp>(operation);
    if (!load)
      continue;
    std::optional<int64_t> offset =
        getConstantOffset(load, loop, accessValueCache);
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
