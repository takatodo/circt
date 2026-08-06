//===- Algorithms.h - Library of scheduling algorithms ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of scheduling algorithms.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_ALGORITHMS_H
#define CIRCT_SCHEDULING_ALGORITHMS_H

#include "circt/Scheduling/Problems.h"

#include <cstdint>

namespace circt {
namespace scheduling {

/// This is a simple list scheduler for solving the basic scheduling problem.
/// Its objective is to assign each operation its earliest possible start time,
/// or in other words, to schedule each operation as soon as possible (hence the
/// name). Fails if the dependence graph contains cycles.
LogicalResult scheduleASAP(Problem &prob);

/// Solve the basic problem using linear programming and a handwritten
/// implementation of the simplex algorithm. The objective is to minimize the
/// start time of the given \p lastOp. Fails if the dependence graph contains
/// cycles, or \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(Problem &prob, Operation *lastOp);

/// Solve the resource-free cyclic problem using linear programming and a
/// handwritten implementation of the simplex algorithm. The objectives are to
/// determine the smallest feasible initiation interval, and to minimize the
/// start time of the given \p lastOp. Fails if the dependence graph contains
/// cycles that do not include at least one edge with a non-zero distance, or
/// \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(CyclicProblem &prob, Operation *lastOp);

/// Solve the acyclic problem with shared operators using a linear
/// programming-based heuristic. The approach tries to minimize the start time
/// of the given \p lastOp, but optimality is not guaranteed. Fails if the
/// dependence graph contains cycles, or \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(SharedOperatorsProblem &prob, Operation *lastOp);

/// Solve the modulo scheduling problem using a linear programming-based
/// heuristic. The approach tries to determine the smallest feasible initiation
/// interval, and to minimize the start time of the given \p lastOp, but
/// optimality is not guaranteed. Fails if the dependence graph contains cycles
/// that do not include at least one edge with a non-zero distance, \p prob
/// does not include \p lastOp, or \p lastOp is not the unique sink of the
/// dependence graph.
LogicalResult scheduleSimplex(ModuloProblem &prob, Operation *lastOp);

/// Solve the acyclic, chaining-enabled problem using linear programming and a
/// handwritten implementation of the simplex algorithm. This approach strictly
/// adheres to the given maximum \p cycleTime. The objective is to minimize the
/// start time of the given \p lastOp. Fails if the dependence graph contains
/// cycles, or individual operator types have delays larger than \p cycleTime,
/// or \p prob does not include \p lastOp.
LogicalResult scheduleSimplex(ChainingProblem &prob, Operation *lastOp,
                              float cycleTime);

/// Solve the resource-free cyclic, chaining-enabled problem using linear
/// programming-based and a handwritten implementation of the simplex algorithm.
/// This approach is an hybrid approach of the ChainingProblem simplex scheduler
/// and the CyclicProblem simplex scheduler. The objectives include determining
/// the smallest feasible initiation interval, and to minimize the start time of
/// a given \p lastOp. Fails if the dependence graph contains cycles that does
/// not include at least one edge with a non-zero distance, individual operator
/// types have delays larger than \p cycleTime, or \p prob does not include
/// \p lastOp.
LogicalResult scheduleSimplex(ChainingCyclicProblem &prob, Operation *lastOp,
                              float cycleTime);

/// Solve the basic problem using linear programming and an external LP solver.
/// The objective is to minimize the start time of the given \p lastOp. Fails if
/// the dependence graph contains cycles, or \p prob does not include \p lastOp.
LogicalResult scheduleLP(Problem &prob, Operation *lastOp);

/// Solve the resource-free cyclic problem using integer linear programming and
/// an external ILP solver. The objectives are to determine the smallest
/// feasible initiation interval, and to minimize the start time of the given \p
/// lastOp. Fails if the dependence graph contains cycles that do not include at
/// least one edge with a non-zero distance, or \p prob does not include
/// \p lastOp.
LogicalResult scheduleLP(CyclicProblem &prob, Operation *lastOp);

/// Result state of a CP-SAT invocation. A feasible result has a proven
/// initiation interval, but may not have a latency-optimal schedule when a
/// time limit expires.
enum class CPSATSolveStatus { optimal, feasible, infeasible, unknown };

/// Resource encoding used by the exact modulo scheduler.
enum class CPSATModuloResourceModel { autoSelect, oneHot, cumulative };

/// Configuration for a CP-SAT scheduler invocation. A zero time limit leaves
/// the search unbounded.
struct CPSATSchedulerOptions {
  double timeLimitSeconds = 0.0;
  /// Zero lets OR-Tools choose its default worker count.
  unsigned numWorkers = 0;
  /// Whether to optimize the designated last operation after minimizing II.
  bool minimizeLatency = true;
  /// Try a restricted balanced start-count model before the complete modulo
  /// resource model. A failed restricted attempt always falls back. During
  /// latency minimization, a verified candidate becomes a complete-model hint
  /// and a safe objective upper bound.
  bool enableBalancedProbe = false;
  /// Per-candidate wall-clock cap for the restricted probe.
  double balancedProbeTimeLimitSeconds = 1.0;
  /// Auto-select avoids a large task/phase/hold Boolean expansion.
  CPSATModuloResourceModel moduloResourceModel =
      CPSATModuloResourceModel::autoSelect;
};

/// Result metadata for a CP-SAT invocation.
struct CPSATSchedulerResult {
  CPSATSolveStatus status = CPSATSolveStatus::unknown;
  /// First candidate II after resource-demand and cyclic-dependence bounds.
  unsigned lowerBound = 0;
  unsigned initiationInterval = 0;
  /// Effective modulo resource encoding and its avoided one-hot size.
  CPSATModuloResourceModel moduloResourceModel =
      CPSATModuloResourceModel::autoSelect;
  uint64_t phaseIndicatorCount = 0;
  bool balancedProbeAttempted = false;
  bool balancedProbeSucceeded = false;
};

/// Solve the acyclic problem with shared operators using constraint programming
/// and an external SAT solver. The objective is to minimize the start time of
/// the given \p lastOp. Fails if the dependence graph contains cycles, or \p
/// prob does not include \p lastOp.
LogicalResult scheduleCPSAT(SharedOperatorsProblem &prob, Operation *lastOp,
                            const CPSATSchedulerOptions &options = {},
                            CPSATSchedulerResult *result = nullptr);

/// Solve the modulo scheduling problem using constraint programming and an
/// external SAT solver. The scheduler enumerates initiation intervals in
/// increasing order and optionally minimizes the start time of \p lastOp at
/// the first feasible candidate while enforcing cyclic dependences and modulo
/// resource reservations. Fails if no feasible schedule is found, or \p
/// lastOp is not part of the problem.
LogicalResult scheduleCPSAT(ModuloProblem &prob, Operation *lastOp,
                            const CPSATSchedulerOptions &options = {},
                            CPSATSchedulerResult *result = nullptr);

/// Configuration for a bounded CP-SAT improvement of an existing feasible
/// modulo schedule. The neighborhood contains the objective's tight fan-in
/// and operations competing with it for limited resources. Operations outside
/// the neighborhood retain their current start times and reservations.
struct CPSATModuloLocalSearchOptions {
  unsigned neighborhoodSize = 32;
  double timeLimitSeconds = 0.1;
  uint64_t seed = 0;
};

/// Result metadata for a bounded modulo local-search invocation.
struct CPSATModuloLocalSearchResult {
  CPSATSolveStatus status = CPSATSolveStatus::unknown;
  unsigned neighborhoodSize = 0;
  unsigned initialObjective = 0;
  unsigned finalObjective = 0;
};

/// Improve an existing feasible modulo schedule at its current initiation
/// interval by re-optimizing a bounded neighborhood with CP-SAT. The incumbent
/// schedule is retained if the time limit expires or no improvement is found.
LogicalResult
improveModuloScheduleCPSAT(ModuloProblem &prob, Operation *lastOp,
                           const CPSATModuloLocalSearchOptions &options = {},
                           CPSATModuloLocalSearchResult *result = nullptr);

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_ALGORITHMS_H
