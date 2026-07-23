//===- AffineSlidingReuseSupport.cpp - Shared reuse analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AffineSlidingReuseSupport.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

using namespace mlir;
using namespace mlir::affine;

namespace circt {
namespace affine_reuse_detail {
namespace {

/// Keep malformed or adversarial affine.apply chains from exhausting the C++
/// stack. Exceeding this proof limit only makes callers conservative.
constexpr unsigned maxAffineExpansionDepth = 256;

/// An exact linear expression over SSA leaf values. APInt widths grow with the
/// expression so composing affine.apply chains cannot wrap at int64_t.
constexpr unsigned maxExactLinearExpansionDepth = 256;

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
  if (depth > maxExactLinearExpansionDepth)
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
  if (depth > maxExactLinearExpansionDepth)
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

static void sortInvariantCoefficients(CheckedLinearExpr &expression) {
  llvm::sort(
      expression.invariantCoefficients, [](const auto &lhs, const auto &rhs) {
        return lhs.first.getAsOpaquePointer() < rhs.first.getAsOpaquePointer();
      });
}

static std::optional<CheckedLinearExpr>
addCheckedLinearExpr(const CheckedLinearExpr &lhs,
                     const CheckedLinearExpr &rhs) {
  CheckedLinearExpr result = lhs;
  if (llvm::AddOverflow(lhs.inductionCoefficient, rhs.inductionCoefficient,
                        result.inductionCoefficient) ||
      llvm::AddOverflow(lhs.constant, rhs.constant, result.constant))
    return std::nullopt;

  for (const auto &termToAdd : rhs.invariantCoefficients) {
    Value value = termToAdd.first;
    int64_t coefficient = termToAdd.second;
    auto existing =
        llvm::find_if(result.invariantCoefficients,
                      [&](const auto &term) { return term.first == value; });
    if (existing == result.invariantCoefficients.end()) {
      result.invariantCoefficients.push_back({value, coefficient});
      continue;
    }
    int64_t sum = 0;
    if (llvm::AddOverflow(existing->second, coefficient, sum))
      return std::nullopt;
    if (sum == 0)
      result.invariantCoefficients.erase(existing);
    else
      existing->second = sum;
  }
  sortInvariantCoefficients(result);
  return result;
}

static std::optional<CheckedLinearExpr>
scaleCheckedLinearExpr(const CheckedLinearExpr &expression, int64_t factor) {
  CheckedLinearExpr result{0, 0, {}};
  if (llvm::MulOverflow(expression.inductionCoefficient, factor,
                        result.inductionCoefficient) ||
      llvm::MulOverflow(expression.constant, factor, result.constant))
    return std::nullopt;
  for (auto [value, coefficient] : expression.invariantCoefficients) {
    int64_t product = 0;
    if (llvm::MulOverflow(coefficient, factor, product))
      return std::nullopt;
    if (product != 0)
      result.invariantCoefficients.push_back({value, product});
  }
  sortInvariantCoefficients(result);
  return result;
}

} // namespace

CheckedLinearExprExpander::CheckedLinearExprExpander(Value inductionVariable)
    : inductionVariable(inductionVariable) {}

std::optional<CheckedLinearExpr>
CheckedLinearExprExpander::expand(AffineExpr expression, ValueRange operands,
                                  unsigned numDims) {
  return expandExpression(expression, operands, numDims, /*depth=*/0);
}

std::optional<CheckedLinearExpr>
CheckedLinearExprExpander::expandValue(Value value, unsigned depth) {
  auto cached = valueCache.find(value);
  if (cached != valueCache.end())
    return cached->second;
  if (depth > maxAffineExpansionDepth)
    return std::nullopt;

  std::optional<CheckedLinearExpr> expansion;
  if (value == inductionVariable) {
    expansion = CheckedLinearExpr{1, 0, {}};
  } else if (IntegerAttr constant; matchPattern(value, m_Constant(&constant))) {
    const llvm::APInt &integer = constant.getValue();
    if (integer.isSignedIntN(64))
      expansion = CheckedLinearExpr{0, integer.getSExtValue(), {}};
  } else if (auto apply = value.getDefiningOp<AffineApplyOp>()) {
    expansion = expandExpression(apply.getAffineMap().getResult(0),
                                 apply.getMapOperands(),
                                 apply.getAffineMap().getNumDims(), depth + 1);
  } else {
    expansion = CheckedLinearExpr{0, 0, {{value, 1}}};
  }
  valueCache.insert({value, expansion});
  return expansion;
}

std::optional<CheckedLinearExpr>
CheckedLinearExprExpander::expandExpression(AffineExpr expression,
                                            ValueRange operands,
                                            unsigned numDims, unsigned depth) {
  if (depth > maxAffineExpansionDepth)
    return std::nullopt;
  if (auto constant = dyn_cast<AffineConstantExpr>(expression))
    return CheckedLinearExpr{0, constant.getValue(), {}};
  if (auto dim = dyn_cast<AffineDimExpr>(expression))
    return expandValue(operands[dim.getPosition()], depth + 1);
  if (auto symbol = dyn_cast<AffineSymbolExpr>(expression))
    return expandValue(operands[numDims + symbol.getPosition()], depth + 1);

  auto binary = dyn_cast<AffineBinaryOpExpr>(expression);
  if (!binary)
    return std::nullopt;
  std::optional<CheckedLinearExpr> lhs =
      expandExpression(binary.getLHS(), operands, numDims, depth + 1);
  std::optional<CheckedLinearExpr> rhs =
      expandExpression(binary.getRHS(), operands, numDims, depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;

  if (expression.getKind() == AffineExprKind::Add)
    return addCheckedLinearExpr(*lhs, *rhs);

  if (expression.getKind() != AffineExprKind::Mul)
    return std::nullopt;

  bool lhsConstant =
      lhs->inductionCoefficient == 0 && lhs->invariantCoefficients.empty();
  bool rhsConstant =
      rhs->inductionCoefficient == 0 && rhs->invariantCoefficients.empty();
  if (!lhsConstant && !rhsConstant)
    return std::nullopt;
  return lhsConstant ? scaleCheckedLinearExpr(*rhs, lhs->constant)
                     : scaleCheckedLinearExpr(*lhs, rhs->constant);
}

std::optional<uint64_t> getExactPositiveTripCount(AffineForOp loop) {
  if (loop.getStepAsInt() != 1)
    return std::nullopt;
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

Operation *findModifyingOperation(AffineForOp loop, Value source,
                                  AliasAnalysis &aliasAnalysis) {
  Operation *modifyingOperation = nullptr;
  (void)loop.getBody()->walk<WalkOrder::PostOrder>([&](Operation *operation) {
    // Operations with recursive memory effects derive their effects from
    // nested operations when they do not also implement a direct effect
    // interface. The post-order walk has already checked those operations.
    if (operation->hasTrait<OpTrait::HasRecursiveMemoryEffects>() &&
        !isa<MemoryEffectOpInterface>(operation))
      return WalkResult::advance();
    if (aliasAnalysis.getModRef(operation, source).isMod()) {
      modifyingOperation = operation;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return modifyingOperation;
}

} // namespace affine_reuse_detail
} // namespace circt
