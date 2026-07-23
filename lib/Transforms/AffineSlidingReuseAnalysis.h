//===- AffineSlidingReuseAnalysis.h - Sliding reuse analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Private analysis shared by affine sliding-window transformations.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_AFFINESLIDINGREUSEANALYSIS_H
#define LIB_TRANSFORMS_AFFINESLIDINGREUSEANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace circt {
namespace affine_reuse_detail {

/// A checked linear expression relative to one loop induction variable:
/// `inductionCoefficient * iv + sum(coefficient * invariant) + constant`.
/// Invariant coefficients are zero-elided and sorted by opaque Value pointer so
/// callers can use them as a structural identity key.
struct CheckedLinearExpr {
  int64_t inductionCoefficient;
  int64_t constant;
  llvm::SmallVector<std::pair<mlir::Value, int64_t>, 4> invariantCoefficients;
};

/// Expand affine expressions relative to one induction variable without
/// overflowing int64_t or recursively following adversarial affine.apply
/// chains without a bound. Addition and multiplication by a proven constant
/// are supported; division and modulo remain conservative failures. Failed
/// proofs are cached conservatively.
class CheckedLinearExprExpander {
public:
  explicit CheckedLinearExprExpander(mlir::Value inductionVariable);

  std::optional<CheckedLinearExpr> expand(mlir::AffineExpr expression,
                                          mlir::ValueRange operands,
                                          unsigned numDims);

private:
  std::optional<CheckedLinearExpr> expandExpression(mlir::AffineExpr expression,
                                                    mlir::ValueRange operands,
                                                    unsigned numDims,
                                                    unsigned depth);
  std::optional<CheckedLinearExpr> expandValue(mlir::Value value,
                                               unsigned depth);

  mlir::Value inductionVariable;
  llvm::DenseMap<mlir::Value, std::optional<CheckedLinearExpr>> valueCache;
};

/// Return true if substituting the loop's lower bound for the induction
/// variable in every result of the load access map is representable without
/// signed 64-bit overflow. Affine.apply operands are composed recursively.
bool isFirstIterationLoadCompositionRepresentable(
    mlir::affine::AffineLoadOp load, mlir::affine::AffineForOp loop,
    CheckedLinearExprExpander &expander);

/// Return the exact positive trip count of a unit-step loop when every upper
/// bound is the lower bound plus a positive constant offset. Bounds and
/// affine.apply chains are compared with arbitrary-width signed arithmetic.
/// Empty, nonlinear, and unsupported bounds conservatively return nullopt.
std::optional<uint64_t>
getExactPositiveTripCount(mlir::affine::AffineForOp loop);

/// Return true if an operation in the loop may modify the source. Unknown
/// effects intentionally block transformations.
bool mayModifySource(mlir::affine::AffineForOp loop, mlir::Value source,
                     mlir::AliasAnalysis &aliasAnalysis);

} // namespace affine_reuse_detail
} // namespace circt

#endif // LIB_TRANSFORMS_AFFINESLIDINGREUSEANALYSIS_H
