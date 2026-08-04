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

/// Solve the resource-free cyclic, chaining-enabled problem using a linear
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

/// Configuration for a CP-SAT scheduler invocation. A zero time limit leaves
/// the search unbounded.
struct CPSATSchedulerOptions {
  double timeLimitSeconds = 0.0;
};

/// Result metadata for a CP-SAT invocation.
struct CPSATSchedulerResult {
  CPSATSolveStatus status = CPSATSolveStatus::unknown;
  /// First candidate II after resource-demand and cyclic-dependence bounds.
  unsigned lowerBound = 0;
  unsigned initiationInterval = 0;
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
/// increasing order and, for each candidate, optimally minimizes the start
/// time of \p lastOp while enforcing cyclic dependences and modulo resource
/// reservations. Fails if no feasible schedule is found, or \p lastOp is not
/// part of the problem.
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

/// Configuration for the node-level reinforcement-learning scheduler.
///
/// The scheduler trains a lightweight linear policy directly on the problem
/// instance. For acyclic problems the policy selects one ready operation from
/// the dependence graph. For modulo problems it selects an ordering between
/// operations whose periodic resource reservations conflict. The terminal
/// reward is based on the start time of `lastOp`. A fixed seed makes training
/// reproducible.
struct NodeRLSchedulerOptions {
  /// Maximum number of policy-training episodes per candidate II.
  unsigned episodes = 64;

  /// Stop a modulo search after this many node-episodes pass without improving
  /// the best result. The effective patience is this budget divided by the
  /// graph's node count, with a minimum of eight episodes. Zero disables
  /// adaptive stopping.
  unsigned episodeNodeBudget = 65536;

  /// Maximum number of resource-ordering constraints considered across all
  /// candidate IIs and episodes. Reaching the limit stops training when a
  /// feasible result is already available and otherwise reports failure. Zero
  /// disables the limit.
  unsigned resourceOrderingBudget = 1048576;

  /// Re-optimize this many operations around the modulo objective with a
  /// bounded CP-SAT neighborhood after NodeRL finds its smallest feasible II.
  /// Zero disables local search. This requires an OR-Tools-enabled build.
  unsigned localSearchNodes = 0;

  /// Wall-clock limit for the optional CP-SAT neighborhood improvement.
  double localSearchTimeLimitSeconds = 0.1;

  /// Seed for sampling actions from the policy.
  uint64_t seed = 0;

  /// Step size used by the REINFORCE policy update.
  double learningRate = 0.05;

  /// Initial softmax temperature.  Higher values explore more actions.
  double exploration = 1.0;
};

/// A requested number of instances for one resource type.
struct ResourceLimit {
  Problem::ResourceType resource;
  unsigned limit;
};

/// A complete allocation of all resource types in a problem.
struct ResourceAllocation {
  SmallVector<ResourceLimit, 4> limits;
};

/// A non-dominated resource allocation discovered by an outer exploration.
struct ResourceParetoPoint {
  ResourceAllocation allocation;
  unsigned latency;
  uint64_t resourceCost;
  /// Pipeline II for a modulo schedule, or zero for an acyclic schedule.
  unsigned initiationInterval = 0;
};

/// Computes the implementation cost of a complete resource allocation.
using ResourceCostFunction = function_ref<uint64_t(const ResourceAllocation &)>;

/// Solve an acyclic resource-constrained problem using node-level
/// reinforcement learning.  The scheduler learns a priority policy over ready
/// operations and retains the best feasible schedule seen during training.  It
/// supports operations linked to multiple limited resources.
///
/// The objective is to minimize the start time of \p lastOp.  Fails if the
/// dependence graph contains cycles, \p prob does not include \p lastOp, or the
/// options are invalid.
LogicalResult scheduleNodeRL(SharedOperatorsProblem &prob, Operation *lastOp,
                             const NodeRLSchedulerOptions &options = {});

/// Solve a modulo scheduling problem using node-level reinforcement learning.
/// The scheduler searches increasing initiation intervals, satisfies loop-
/// carried dependences as cyclic difference constraints, and resolves resource
/// conflicts with a modulo reservation table.  It supports multiple resources
/// per operation and resource initiation intervals.  The returned schedule has
/// the smallest feasible II encountered; among policies at that II it retains
/// the one with the earliest \p lastOp.
LogicalResult scheduleNodeRL(ModuloProblem &prob, Operation *lastOp,
                             const NodeRLSchedulerOptions &options = {});

/// Schedule each complete resource allocation with NodeRL and return the
/// latency/resource-cost Pareto frontier. A resource type's cost defaults to
/// one if it is not set. The problem is left scheduled using the least-cost
/// point on the returned frontier (breaking ties by latency).
///
/// Every allocation must specify each resource type exactly once, and each
/// requested limit must be non-zero. Fails if an allocation or its schedule is
/// invalid.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreNodeRLPareto(SharedOperatorsProblem &prob, Operation *lastOp,
                    ArrayRef<ResourceAllocation> allocations,
                    const NodeRLSchedulerOptions &options = {});

/// As above, but use a client-provided cost model. This permits clients to use
/// non-linear or post-synthesis resource-cost estimates.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreNodeRLPareto(SharedOperatorsProblem &prob, Operation *lastOp,
                    ArrayRef<ResourceAllocation> allocations,
                    ResourceCostFunction costFunction,
                    const NodeRLSchedulerOptions &options = {});

/// Modulo-scheduling variant of the NodeRL Pareto exploration. Each candidate
/// is scheduled with cyclic resource constraints; its reported latency is the
/// start time of \p lastOp and its II is included in the returned point and
/// stored in the problem left behind by the selected frontier point.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreNodeRLPareto(ModuloProblem &prob, Operation *lastOp,
                    ArrayRef<ResourceAllocation> allocations,
                    const NodeRLSchedulerOptions &options = {});

/// As above, but use a client-provided cost model, for example measured
/// post-synthesis area of an allocation.
FailureOr<SmallVector<ResourceParetoPoint>>
exploreNodeRLPareto(ModuloProblem &prob, Operation *lastOp,
                    ArrayRef<ResourceAllocation> allocations,
                    ResourceCostFunction costFunction,
                    const NodeRLSchedulerOptions &options = {});

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_ALGORITHMS_H
