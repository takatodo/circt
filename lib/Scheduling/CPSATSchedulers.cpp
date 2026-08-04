//===- CPSATSchedulers.cpp - Schedulers using external CPSAT solvers
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of cp-sat programming-based schedulers using external solvers
// via OR-Tools.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Operation.h"

#include "ortools/sat/cp_model.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"

#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

#define DEBUG_TYPE "cpsat-schedulers"

using namespace circt;
using namespace circt::scheduling;
using namespace operations_research;
using namespace operations_research::sat;

using llvm::dbgs;
using llvm::format;

namespace {

/// Select a deterministic local-search neighborhood around the requested
/// objective. Roughly half of the budget follows the tightest incoming
/// dependences. The remainder admits operations that compete with that fan-in
/// for limited resources, which lets CP-SAT coordinate precedence and phase
/// changes without fixing an arbitrary resource partition up front.
static SmallVector<Operation *> selectModuloNeighborhood(ModuloProblem &prob,
                                                         Operation *lastOp,
                                                         unsigned maxNodes) {
  auto &tasks = prob.getOperations();
  maxNodes = std::min<unsigned>(maxNodes, tasks.size());
  if (maxNodes == tasks.size())
    return SmallVector<Operation *>(tasks.begin(), tasks.end());

  DenseMap<Operation *, unsigned> taskIndices;
  SmallVector<int64_t> startTimes;
  startTimes.reserve(tasks.size());
  for (auto [index, task] : llvm::enumerate(tasks)) {
    taskIndices[task] = index;
    startTimes.push_back(*prob.getStartTime(task));
  }
  unsigned ii = *prob.getInitiationInterval();
  unsigned lastNode = taskIndices.lookup(lastOp);

  using DependenceCandidate = std::tuple<int64_t, unsigned, unsigned>;
  std::priority_queue<DependenceCandidate, std::vector<DependenceCandidate>,
                      std::greater<DependenceCandidate>>
      dependenceCandidates;
  auto pushPredecessors = [&](unsigned destination) {
    for (auto dependence : prob.getDependences(tasks[destination])) {
      unsigned source = taskIndices.lookup(dependence.getSource());
      int64_t latency =
          *prob.getLatency(*prob.getLinkedOperatorType(tasks[source]));
      int64_t distance = prob.getDistance(dependence).value_or(0);
      int64_t required =
          startTimes[source] + latency - distance * static_cast<int64_t>(ii);
      int64_t slack = startTimes[destination] - required;
      assert(slack >= 0 && "incumbent must satisfy all dependences");
      dependenceCandidates.emplace(slack, source, destination);
    }
  };

  SmallVector<char> selected(tasks.size(), false);
  SmallVector<unsigned> neighborhood{lastNode};
  selected[lastNode] = true;
  pushPredecessors(lastNode);
  unsigned criticalBudget = std::max(1u, (maxNodes + 1) / 2);
  while (neighborhood.size() < criticalBudget &&
         !dependenceCandidates.empty()) {
    auto [slack, source, destination] = dependenceCandidates.top();
    (void)slack;
    (void)destination;
    dependenceCandidates.pop();
    if (selected[source])
      continue;
    selected[source] = true;
    neighborhood.push_back(source);
    pushPredecessors(source);
  }

  DenseMap<Problem::ResourceType, SmallVector<unsigned, 4>> resourceUsers;
  for (auto [node, task] : llvm::enumerate(tasks)) {
    auto resources = prob.getLinkedResourceTypes(task);
    if (!resources)
      continue;
    for (auto resource : *resources)
      if (prob.getLimit(resource).value_or(0) > 0)
        resourceUsers[resource].push_back(node);
  }

  SmallVector<unsigned> bestResourceDistance(
      tasks.size(), std::numeric_limits<unsigned>::max());
  DenseMap<Problem::ResourceType, SmallVector<unsigned, 4>> resourceAnchors;
  unsigned criticalNodes = neighborhood.size();
  for (unsigned index = 0; index != criticalNodes; ++index) {
    unsigned anchor = neighborhood[index];
    auto resources = prob.getLinkedResourceTypes(tasks[anchor]);
    if (!resources)
      continue;
    for (auto resource : *resources) {
      if (prob.getLimit(resource).value_or(0) == 0)
        continue;
      resourceAnchors[resource].push_back(anchor);
    }
  }
  for (const auto &[resource, anchors] : resourceAnchors) {
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    for (unsigned peer : resourceUsers.lookup(resource)) {
      if (selected[peer])
        continue;
      for (unsigned anchor : anchors) {
        unsigned anchorPhase = static_cast<uint64_t>(startTimes[anchor]) % ii;
        unsigned peerPhase = static_cast<uint64_t>(startTimes[peer]) % ii;
        unsigned forward = (anchorPhase + ii - peerPhase) % ii;
        unsigned backward = (peerPhase + ii - anchorPhase) % ii;
        unsigned distance = std::min(forward, backward);
        distance = distance > resourceII ? distance - resourceII : 0;
        bestResourceDistance[peer] =
            std::min(bestResourceDistance[peer], distance);
      }
    }
  }

  SmallVector<std::pair<unsigned, unsigned>> resourceCandidates;
  for (unsigned node = 0, e = tasks.size(); node != e; ++node)
    if (!selected[node] &&
        bestResourceDistance[node] != std::numeric_limits<unsigned>::max())
      resourceCandidates.emplace_back(bestResourceDistance[node], node);
  llvm::sort(resourceCandidates);
  for (auto [distance, node] : resourceCandidates) {
    (void)distance;
    if (neighborhood.size() == maxNodes)
      break;
    selected[node] = true;
    neighborhood.push_back(node);
  }

  // If the objective does not use enough limited resources to fill the
  // neighborhood, continue through its dependence fan-in.
  while (neighborhood.size() < maxNodes && !dependenceCandidates.empty()) {
    auto [slack, source, destination] = dependenceCandidates.top();
    (void)slack;
    (void)destination;
    dependenceCandidates.pop();
    if (selected[source])
      continue;
    selected[source] = true;
    neighborhood.push_back(source);
    pushPredecessors(source);
  }

  llvm::sort(neighborhood);
  SmallVector<Operation *> result;
  result.reserve(neighborhood.size());
  for (unsigned node : neighborhood)
    result.push_back(tasks[node]);
  return result;
}

/// Compute the first integer II that can satisfy the schedule-independent
/// resource demand and all cyclic dependences. Resource conflicts can require
/// a larger II, but no CP-SAT model below this bound could be feasible.
static unsigned computeModuloLowerBound(ModuloProblem &prob,
                                        unsigned upperBound) {
  auto &tasks = prob.getOperations();
  unsigned lowerBound = 1;
  for (auto resource : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(resource).value_or(0);
    if (limit == 0)
      continue;
    uint64_t demand = 0;
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    for (Operation *task : tasks) {
      auto resources = prob.getLinkedResourceTypes(task);
      if (resources && llvm::is_contained(*resources, resource))
        demand += resourceII;
    }
    lowerBound = std::max<uint64_t>(
        lowerBound, (demand + limit - 1) / static_cast<uint64_t>(limit));
  }

  DenseMap<Operation *, unsigned> indices;
  for (auto [index, task] : llvm::enumerate(tasks))
    indices[task] = index;
  auto dependencesFeasible = [&](unsigned ii) {
    SmallVector<int64_t> starts(tasks.size(), 0);
    for (unsigned iteration = 0; iteration != tasks.size(); ++iteration) {
      bool changed = false;
      for (Operation *destination : tasks) {
        for (auto dependence : prob.getDependences(destination)) {
          Operation *source = dependence.getSource();
          int64_t latency =
              *prob.getLatency(*prob.getLinkedOperatorType(source));
          int64_t distance = prob.getDistance(dependence).value_or(0);
          int64_t required = starts[indices.lookup(source)] + latency -
                             distance * static_cast<int64_t>(ii);
          auto &destinationStart = starts[indices.lookup(destination)];
          if (required <= destinationStart)
            continue;
          destinationStart = required;
          changed = true;
        }
      }
      if (!changed)
        return true;
      if (iteration + 1 == tasks.size())
        return false;
    }
    return true;
  };

  while (lowerBound <= upperBound && !dependencesFeasible(lowerBound))
    ++lowerBound;
  return lowerBound;
}

/// A fully serial schedule waits long enough for every resource used by an
/// operation to accept its next request. Summing these spans is a conservative
/// horizon even when a resource hold is longer than the operation latency.
static unsigned getSerialSpan(SharedOperatorsProblem &prob, Operation *task) {
  unsigned span = *prob.getLatency(*prob.getLinkedOperatorType(task));
  if (auto resources = prob.getLinkedResourceTypes(task))
    for (auto resource : *resources)
      if (prob.getLimit(resource).value_or(0) > 0)
        span = std::max(
            span, prob.getResourceInitiationInterval(resource).value_or(1));
  return span;
}

} // namespace

/// Solve the shared operators problem by modeling it as a Resource
/// Constrained Project Scheduling Problem (RCPSP), which in turn is formulated
/// as a Constraint Programming (CP) Satisfiability (SAT) problem.
///
/// This is a high-fidelity translation of
/// https://github.com/google/or-tools/blob/stable/examples/python/rcpsp_sat.py
/// but a gentler introduction (though with differing formulation) is
/// https://python-mip.readthedocs.io/en/latest/examples.html
LogicalResult scheduling::scheduleCPSAT(SharedOperatorsProblem &prob,
                                        Operation *lastOp,
                                        const CPSATSchedulerOptions &options,
                                        CPSATSchedulerResult *result) {
  Operation *containingOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return containingOp->emitError("problem does not include last operation");
  prob.clearResourceBindings();

  CpModelBuilder cpModel;
  auto &tasks = prob.getOperations();

  DenseMap<Operation *, IntVar> taskStarts;
  DenseMap<Operation *, IntVar> taskEnds;
  DenseMap<Problem::ResourceType, SmallVector<IntervalVar, 4>>
      resourcesToTaskIntervals;

  // First get a horizon from a schedule that executes every operation and
  // waits for each held resource before issuing the next one.
  unsigned horizon = 0;
  for (auto *task : tasks)
    horizon += getSerialSpan(prob, task);

  // Build task-interval decision variables, which effectively serve to
  // constrain startVar and endVar to be duration apart. Then map them
  // to the resources (operators) consumed during those intervals. Note,
  // resources are in fact not constrained to be occupied for the whole of the
  // task interval, but only during the first "tick". See comment below
  // regarding cpModel.NewFixedSizeIntervalVar.
  for (auto item : llvm::enumerate(tasks)) {
    auto i = item.index();
    auto *task = item.value();
    IntVar startVar = cpModel.NewIntVar(Domain(0, horizon))
                          .WithName((Twine("start_of_task_") + Twine(i)).str());
    IntVar endVar = cpModel.NewIntVar(Domain(0, horizon))
                        .WithName((Twine("end_of_task_") + Twine(i)).str());
    taskStarts[task] = startVar;
    taskEnds[task] = endVar;
    auto opr = prob.getLinkedOperatorType(task);
    unsigned duration = *prob.getLatency(*opr);
    IntervalVar taskInterval =
        cpModel.NewIntervalVar(startVar, duration, endVar)
            .WithName((Twine("task_interval_") + Twine(i)).str());

    auto resourceListOpt = prob.getLinkedResourceTypes(task);
    if (resourceListOpt) {
      for (const auto &resource : *resourceListOpt) {
        if (auto limitOpt = prob.getLimit(resource); limitOpt && *limitOpt > 0)
          resourcesToTaskIntervals[resource].push_back(taskInterval);
      }
    }
  }

  // Check for cycles and otherwise establish operation ordering
  // constraints.
  for (Operation *task : tasks) {
    for (auto dep : prob.getDependences(task)) {
      Operation *src = dep.getSource();
      Operation *dst = dep.getDestination();
      if (src == dst)
        return containingOp->emitError() << "dependence cycle detected";
      cpModel.AddLessOrEqual(taskEnds[src], taskStarts[dst]);
    }
  }

  // Establish "cumulative" constraints in order to constrain maximum
  // concurrent usage of operators.
  for (auto resourceToTaskIntervals : resourcesToTaskIntervals) {
    Problem::ResourceType &resource = resourceToTaskIntervals.getFirst();
    auto capacity = prob.getLimit(resource);
    SmallVector<IntervalVar, 4> &taskIntervals =
        resourceToTaskIntervals.getSecond();
    // The semantics of cumulative constraints in or-tools are such that
    // for any integer point, the sum of the demands across all
    // intervals containing that point does not exceed the capacity of the
    // resource. Thus tasks, in 1-1 correspondence with their intervals, are
    // constrained to satisfy maximum resource requirements.
    // See https://or.stackexchange.com/a/3363 for more details.
    CumulativeConstraint cumu = cpModel.AddCumulative(capacity.value());
    for (const auto &item : llvm::enumerate(taskIntervals)) {
      auto i = item.index();
      auto taskInterval = item.value();
      IntVar demandVar = cpModel.NewIntVar(Domain(1)).WithName(
          (Twine("demand_") + Twine(i) + Twine("_") +
           Twine(resource.getAttr().strref()))
              .str());
      // Resources are reserved from issue until their next accepted request.
      IntervalVar start = cpModel.NewFixedSizeIntervalVar(
          taskInterval.StartExpr(),
          prob.getResourceInitiationInterval(resource).value_or(1));
      cumu.AddDemand(start, demandVar);
    }
  }

  cpModel.Minimize(taskEnds[lastOp]);

  Model model;
  if (options.timeLimitSeconds > 0.0) {
    SatParameters parameters;
    parameters.set_max_time_in_seconds(options.timeLimitSeconds);
    model.Add(NewSatParameters(parameters));
  }

  int numSolutions = 0;
  model.Add(NewFeasibleSolutionObserver([&](const CpSolverResponse &r) {
    LLVM_DEBUG(dbgs() << "Solution " << numSolutions << '\n');
    LLVM_DEBUG(dbgs() << "Solution status" << r.status() << '\n');
    ++numSolutions;
  }));

  LLVM_DEBUG(dbgs() << "Starting solver\n");
  const CpSolverResponse response = SolveCpModel(cpModel.Build(), &model);

  if (response.status() == CpSolverStatus::OPTIMAL ||
      response.status() == CpSolverStatus::FEASIBLE) {
    if (result)
      result->status = response.status() == CpSolverStatus::OPTIMAL
                           ? CPSATSolveStatus::optimal
                           : CPSATSolveStatus::feasible;
    for (auto *task : tasks)
      prob.setStartTime(task, SolutionIntegerValue(response, taskStarts[task]));

    return success();
  }
  if (result)
    result->status = response.status() == CpSolverStatus::INFEASIBLE
                         ? CPSATSolveStatus::infeasible
                         : CPSATSolveStatus::unknown;
  return containingOp->emitError() << "infeasible";
}

/// Solve a modulo scheduling problem exactly for every candidate II. Resource
/// reservations are represented by the congruence class of each operation's
/// start time. This deliberately mirrors ModuloProblem::verifyUtilization:
/// a reservation with initiation interval R occupies (start + offset) mod II
/// for every offset in [0, R).
LogicalResult scheduling::scheduleCPSAT(ModuloProblem &prob, Operation *lastOp,
                                        const CPSATSchedulerOptions &options,
                                        CPSATSchedulerResult *result) {
  Operation *containingOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return containingOp->emitError("problem does not include last operation");

  auto &tasks = prob.getOperations();
  unsigned horizon = 0;
  for (Operation *task : tasks)
    horizon += getSerialSpan(prob, task);

  // A serial allocation of every limited resource, and the sum of operation
  // latencies for a recurrence, provide finite II bounds. This is conservative
  // but makes the exact solver a useful oracle for the small benchmark cases.
  unsigned upperBound = std::max(1u, horizon);
  for (auto resource : prob.getResourceTypes()) {
    if (prob.getLimit(resource).value_or(0) == 0)
      continue;
    unsigned demand = 0;
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    for (Operation *task : tasks) {
      auto resources = prob.getLinkedResourceTypes(task);
      if (resources && llvm::is_contained(*resources, resource))
        demand += resourceII;
    }
    upperBound = std::max(upperBound, demand);
  }

  unsigned lowerBound = computeModuloLowerBound(prob, upperBound);
  if (result)
    result->lowerBound = lowerBound;

  auto startTime = std::chrono::steady_clock::now();
  for (unsigned ii = lowerBound; ii <= upperBound; ++ii) {
    CpModelBuilder cpModel;
    DenseMap<Operation *, IntVar> starts;
    DenseMap<Operation *, IntVar> phases;
    for (auto item : llvm::enumerate(tasks)) {
      unsigned index = item.index();
      Operation *task = item.value();
      starts[task] =
          cpModel.NewIntVar(Domain(0, horizon))
              .WithName((Twine("start_of_task_") + Twine(index)).str());
      phases[task] =
          cpModel.NewIntVar(Domain(0, ii - 1))
              .WithName((Twine("phase_of_task_") + Twine(index)).str());
      cpModel.AddModuloEquality(phases[task], starts[task], ii);
    }

    for (Operation *task : tasks) {
      for (auto dep : prob.getDependences(task)) {
        Operation *src = dep.getSource();
        Operation *dst = dep.getDestination();
        unsigned latency = *prob.getLatency(*prob.getLinkedOperatorType(src));
        unsigned distance = prob.getDistance(dep).value_or(0);
        cpModel.AddGreaterOrEqual(starts[dst],
                                  starts[src] + latency - distance * ii);
      }
    }

    for (auto resource : prob.getResourceTypes()) {
      auto limit = prob.getLimit(resource);
      if (!limit || *limit == 0)
        continue;
      SmallVector<Operation *> users;
      for (Operation *task : tasks) {
        auto resources = prob.getLinkedResourceTypes(task);
        if (resources && llvm::is_contained(*resources, resource))
          users.push_back(task);
      }
      unsigned resourceII =
          prob.getResourceInitiationInterval(resource).value_or(1);
      for (unsigned phase = 0; phase != ii; ++phase) {
        LinearExpr reservations;
        for (Operation *task : users) {
          for (unsigned offset = 0; offset != resourceII; ++offset) {
            unsigned requiredPhase = (phase + ii - offset % ii) % ii;
            BoolVar usesPhase = cpModel.NewBoolVar();
            cpModel.AddEquality(phases[task], requiredPhase)
                .OnlyEnforceIf(usesPhase);
            cpModel.AddNotEqual(phases[task], requiredPhase)
                .OnlyEnforceIf(~usesPhase);
            reservations += usesPhase;
          }
        }
        cpModel.AddLessOrEqual(reservations, *limit);
      }
    }

    cpModel.Minimize(starts[lastOp]);
    Model model;
    if (options.timeLimitSeconds > 0.0) {
      double elapsed = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - startTime)
                           .count();
      double remaining = options.timeLimitSeconds - elapsed;
      if (remaining <= 0.0) {
        if (result)
          result->status = CPSATSolveStatus::unknown;
        return containingOp->emitError("CP-SAT time limit exceeded");
      }
      SatParameters parameters;
      parameters.set_max_time_in_seconds(remaining);
      model.Add(NewSatParameters(parameters));
    }
    const CpSolverResponse response = SolveCpModel(cpModel.Build(), &model);
    if (response.status() == CpSolverStatus::INFEASIBLE)
      continue;
    if (response.status() != CpSolverStatus::OPTIMAL &&
        response.status() != CpSolverStatus::FEASIBLE) {
      if (result)
        result->status = CPSATSolveStatus::unknown;
      return containingOp->emitError("CP-SAT time limit exceeded");
    }

    prob.clearResourceBindings();
    for (Operation *task : tasks)
      prob.setStartTime(task, SolutionIntegerValue(response, starts[task]));
    prob.setInitiationInterval(ii);
    if (result) {
      result->status = response.status() == CpSolverStatus::OPTIMAL
                           ? CPSATSolveStatus::optimal
                           : CPSATSolveStatus::feasible;
      result->initiationInterval = ii;
    }
    if (failed(prob.verify()))
      return failure();
    return success();
  }
  if (result)
    result->status = CPSATSolveStatus::infeasible;
  return containingOp->emitError() << "infeasible";
}

LogicalResult scheduling::improveModuloScheduleCPSAT(
    ModuloProblem &prob, Operation *lastOp,
    const CPSATModuloLocalSearchOptions &options,
    CPSATModuloLocalSearchResult *result) {
  Operation *containingOp = prob.getContainingOp();
  if (!prob.hasOperation(lastOp))
    return containingOp->emitError("problem does not include last operation");
  if (options.neighborhoodSize == 0)
    return containingOp->emitError(
        "CP-SAT modulo local search requires a non-empty neighborhood");
  if (!std::isfinite(options.timeLimitSeconds) ||
      options.timeLimitSeconds <= 0.0)
    return containingOp->emitError(
        "CP-SAT modulo local-search time limit must be finite and positive");
  if (!prob.getInitiationInterval())
    return containingOp->emitError(
        "CP-SAT modulo local search requires an existing initiation interval");
  for (Operation *task : prob.getOperations())
    if (!prob.getStartTime(task))
      return containingOp->emitError(
          "CP-SAT modulo local search requires an existing schedule");
  if (failed(prob.verify()))
    return failure();

  unsigned ii = *prob.getInitiationInterval();
  unsigned initialObjective = *prob.getStartTime(lastOp);
  if (result) {
    result->initialObjective = initialObjective;
    result->finalObjective = initialObjective;
  }

  SmallVector<Operation *> neighborhood =
      selectModuloNeighborhood(prob, lastOp, options.neighborhoodSize);
  if (result)
    result->neighborhoodSize = neighborhood.size();
  DenseSet<Operation *> localTasks(neighborhood.begin(), neighborhood.end());

  DenseMap<Operation *, int64_t> incumbentStarts;
  int64_t maxIncumbentStart = 0;
  for (Operation *task : prob.getOperations()) {
    int64_t start = *prob.getStartTime(task);
    incumbentStarts[task] = start;
    maxIncumbentStart = std::max(maxIncumbentStart, start);
  }

  int64_t extension = ii;
  for (Operation *task : neighborhood) {
    unsigned span = *prob.getLatency(*prob.getLinkedOperatorType(task));
    if (auto resources = prob.getLinkedResourceTypes(task))
      for (auto resource : *resources)
        span = std::max(
            span, prob.getResourceInitiationInterval(resource).value_or(1));
    extension += span;
  }
  int64_t horizon = maxIncumbentStart + extension;

  CpModelBuilder cpModel;
  DenseMap<Operation *, IntVar> starts;
  DenseMap<Operation *, IntVar> phases;
  for (auto [index, task] : llvm::enumerate(neighborhood)) {
    starts[task] = cpModel.NewIntVar(Domain(0, horizon))
                       .WithName((Twine("local_start_") + Twine(index)).str());
    phases[task] = cpModel.NewIntVar(Domain(0, ii - 1))
                       .WithName((Twine("local_phase_") + Twine(index)).str());
    cpModel.AddModuloEquality(phases[task], starts[task], ii);
    cpModel.AddHint(starts[task], incumbentStarts.lookup(task));
    cpModel.AddHint(phases[task], incumbentStarts.lookup(task) % ii);
  }

  // Retain all dependences crossing the neighborhood boundary. An incoming
  // edge becomes a release bound; an outgoing edge becomes a deadline.
  for (Operation *destination : prob.getOperations()) {
    bool localDestination = localTasks.contains(destination);
    for (auto dependence : prob.getDependences(destination)) {
      Operation *source = dependence.getSource();
      bool localSource = localTasks.contains(source);
      if (!localSource && !localDestination)
        continue;
      int64_t latency = *prob.getLatency(*prob.getLinkedOperatorType(source));
      int64_t distance = prob.getDistance(dependence).value_or(0);
      int64_t delay = latency - distance * static_cast<int64_t>(ii);
      if (localSource && localDestination)
        cpModel.AddGreaterOrEqual(starts[destination], starts[source] + delay);
      else if (localDestination)
        cpModel.AddGreaterOrEqual(starts[destination],
                                  incumbentStarts.lookup(source) + delay);
      else
        cpModel.AddLessOrEqual(starts[source] + delay,
                               incumbentStarts.lookup(destination));
    }
  }

  // Subtract the fixed exterior reservations from each phase's capacity, then
  // model only the variable neighborhood users. This keeps the CP model
  // proportional to neighborhood size rather than total graph size.
  for (auto resource : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(resource).value_or(0);
    if (limit == 0)
      continue;
    SmallVector<Operation *> localUsers;
    SmallVector<int64_t> remainingCapacity(ii, limit);
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    for (Operation *task : prob.getOperations()) {
      auto resources = prob.getLinkedResourceTypes(task);
      if (!resources || !llvm::is_contained(*resources, resource))
        continue;
      if (localTasks.contains(task)) {
        localUsers.push_back(task);
        continue;
      }
      unsigned phase = incumbentStarts.lookup(task) % ii;
      for (unsigned offset = 0; offset != resourceII; ++offset) {
        int64_t &capacity = remainingCapacity[(phase + offset) % ii];
        --capacity;
        if (capacity < 0)
          return containingOp->emitError(
              "incumbent overuses a resource during CP-SAT local search");
      }
    }
    if (localUsers.empty())
      continue;
    for (unsigned phase = 0; phase != ii; ++phase) {
      LinearExpr reservations;
      for (Operation *task : localUsers) {
        for (unsigned offset = 0; offset != resourceII; ++offset) {
          unsigned requiredPhase = (phase + ii - offset % ii) % ii;
          BoolVar usesPhase = cpModel.NewBoolVar();
          cpModel.AddEquality(phases[task], requiredPhase)
              .OnlyEnforceIf(usesPhase);
          cpModel.AddNotEqual(phases[task], requiredPhase)
              .OnlyEnforceIf(~usesPhase);
          reservations += usesPhase;
        }
      }
      cpModel.AddLessOrEqual(reservations, remainingCapacity[phase]);
    }
  }

  cpModel.AddLessOrEqual(starts[lastOp], initialObjective);
  cpModel.Minimize(starts[lastOp]);

  SatParameters parameters;
  parameters.set_max_time_in_seconds(options.timeLimitSeconds);
  parameters.set_num_search_workers(1);
  parameters.set_random_seed(
      static_cast<int32_t>(options.seed & std::numeric_limits<int32_t>::max()));
  Model model;
  model.Add(NewSatParameters(parameters));
  const CpSolverResponse response = SolveCpModel(cpModel.Build(), &model);
  if (result)
    result->status = response.status() == CpSolverStatus::OPTIMAL
                         ? CPSATSolveStatus::optimal
                     : response.status() == CpSolverStatus::FEASIBLE
                         ? CPSATSolveStatus::feasible
                     : response.status() == CpSolverStatus::INFEASIBLE
                         ? CPSATSolveStatus::infeasible
                         : CPSATSolveStatus::unknown;
  if (response.status() != CpSolverStatus::OPTIMAL &&
      response.status() != CpSolverStatus::FEASIBLE)
    return success();

  unsigned candidateObjective = SolutionIntegerValue(response, starts[lastOp]);
  if (candidateObjective >= initialObjective)
    return success();

  for (Operation *task : neighborhood)
    prob.setStartTime(task, SolutionIntegerValue(response, starts[task]));
  prob.clearResourceBindings();
  if (failed(prob.verify())) {
    for (Operation *task : neighborhood)
      prob.setStartTime(task, incumbentStarts.lookup(task));
    return containingOp->emitError(
        "CP-SAT local search produced an invalid modulo schedule");
  }
  if (result)
    result->finalObjective = candidateObjective;
  return success();
}
