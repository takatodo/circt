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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

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

/// A requested number of instances for one resource type.
struct ResourceLimit {
  Problem::ResourceType resource;
  unsigned limit;
};

/// A complete allocation of all resource types in a problem.
struct ResourceAllocation {
  SmallVector<ResourceLimit, 4> limits;
};

/// One periodic resource reservation made by an operation in a modulo
/// schedule. The operation issues in `phase` modulo `period` and occupies one
/// of `instances` physical resources for `hold` cycles. Unlike a static
/// binding, the selected physical instance may rotate between iterations.
struct RotatingResourceReservation {
  Problem::ResourceType resource;
  unsigned phase;
  unsigned period;
  unsigned hold;
  unsigned instances;
};

/// The schedule assigned to one operation in problem insertion order.
struct OperationScheduleCertificate {
  unsigned startTime;

  /// Static instance assignments ordered like the operation's limited
  /// resource types. This is absent when the operation uses rotating
  /// reservations instead.
  std::optional<SmallVector<unsigned, 2>> resourceBindings;

  /// Periodic reservations used when no static resource binding exists.
  SmallVector<RotatingResourceReservation, 2> rotatingReservations;
};

/// A self-contained schedule which can be reapplied without invoking a
/// scheduler. Entries correspond one-to-one with `Problem::getOperations()`.
struct ResourceScheduleCertificate {
  SmallVector<OperationScheduleCertificate, 0> operations;
};

/// A non-dominated resource allocation discovered by an outer exploration.
struct ResourceParetoPoint {
  ResourceAllocation allocation;
  unsigned latency;
  uint64_t resourceCost;
  /// Pipeline II for a modulo schedule, or zero for an acyclic schedule.
  unsigned initiationInterval = 0;
  ResourceScheduleCertificate schedule;
};

/// Apply and verify a previously captured Pareto point without rerunning its
/// scheduler. The certificate must describe the operations in problem
/// insertion order and its recorded latency must match `lastOp`'s start time.
LogicalResult applyResourceParetoPoint(SharedOperatorsProblem &prob,
                                       Operation *lastOp,
                                       const ResourceParetoPoint &point);

/// Modulo-scheduling variant. In addition to static resource bindings, this
/// accepts explicit rotating reservations and restores the recorded II.
LogicalResult applyResourceParetoPoint(ModuloProblem &prob, Operation *lastOp,
                                       const ResourceParetoPoint &point);

/// Computes the implementation cost of a complete resource allocation.
using ResourceCostFunction = function_ref<uint64_t(const ResourceAllocation &)>;

/// Schedule each complete resource allocation with CP-SAT and return the
/// latency/resource-cost Pareto frontier. Missing acyclic bindings are
/// synthesized deterministically from the verified interval schedule. The
/// problem is left at the least-cost frontier point without rerunning CP-SAT.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreCPSATPareto(SharedOperatorsProblem &prob, Operation *lastOp,
                   ArrayRef<ResourceAllocation> allocations,
                   const CPSATSchedulerOptions &options = {});

/// As above, but use a client-provided resource cost model.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreCPSATPareto(SharedOperatorsProblem &prob, Operation *lastOp,
                   ArrayRef<ResourceAllocation> allocations,
                   ResourceCostFunction costFunction,
                   const CPSATSchedulerOptions &options = {});

/// Modulo-scheduling CP-SAT Pareto exploration. Unbound resource uses receive
/// a static circular coloring when one is found cheaply, and otherwise are
/// captured as explicit rotating reservations at the solver's selected II.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreCPSATPareto(ModuloProblem &prob, Operation *lastOp,
                   ArrayRef<ResourceAllocation> allocations,
                   const CPSATSchedulerOptions &options = {});

/// As above, but use a client-provided resource cost model.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreCPSATPareto(ModuloProblem &prob, Operation *lastOp,
                   ArrayRef<ResourceAllocation> allocations,
                   ResourceCostFunction costFunction,
                   const CPSATSchedulerOptions &options = {});

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_ALGORITHMS_H
