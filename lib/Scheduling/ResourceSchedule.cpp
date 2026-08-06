//===- ResourceSchedule.cpp - Resource schedule certificates --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ResourceSchedule.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"

#include <tuple>

using namespace circt;
using namespace circt::scheduling;

namespace {

static LogicalResult
validateResourceAllocation(SharedOperatorsProblem &prob,
                           const ResourceAllocation &allocation,
                           DenseMap<Problem::ResourceType, unsigned> &limits) {
  limits.clear();
  for (const auto &entry : allocation.limits) {
    if (!prob.hasResourceType(entry.resource))
      return prob.getContainingOp()->emitError()
             << "resource allocation refers to an unregistered resource '"
             << entry.resource.getValue() << "'";
    if (!limits.try_emplace(entry.resource, entry.limit).second)
      return prob.getContainingOp()->emitError()
             << "resource allocation specifies resource '"
             << entry.resource.getValue() << "' more than once";
    if (entry.limit == 0)
      return prob.getContainingOp()->emitError()
             << "resource allocation has a zero limit for resource '"
             << entry.resource.getValue() << "'";
  }
  if (limits.size() != prob.getResourceTypes().size())
    return prob.getContainingOp()->emitError(
        "resource allocation must specify every resource type exactly once");
  return success();
}

static void applyResourceAllocation(SharedOperatorsProblem &prob,
                                    const ResourceAllocation &allocation) {
  for (const auto &entry : allocation.limits)
    prob.setLimit(entry.resource, entry.limit);
}

static SmallVector<Problem::ResourceType, 2>
getLimitedResources(SharedOperatorsProblem &prob, Operation *op,
                    const DenseMap<Problem::ResourceType, unsigned> &limits) {
  SmallVector<Problem::ResourceType, 2> limitedResources;
  if (auto resources = prob.getLinkedResourceTypes(op))
    for (auto resource : *resources)
      if (limits.lookup(resource) > 0 &&
          !llvm::is_contained(limitedResources, resource))
        limitedResources.push_back(resource);
  return limitedResources;
}

struct ResourceUse {
  unsigned startTime;
  unsigned operationIndex;
  unsigned resourceIndex;
};

/// Deterministically color an acyclic verified interval schedule. Interval
/// graphs are perfect, so the first available physical instance succeeds
/// whenever the problem's utilization verifier accepts the schedule.
static LogicalResult synthesizeAcyclicBindings(
    SharedOperatorsProblem &prob,
    ArrayRef<SmallVector<Problem::ResourceType, 2>> limitedResources,
    ResourceScheduleCertificate &schedule) {
  SmallVector<bool> hadBindings;
  hadBindings.reserve(schedule.operations.size());
  for (const auto &operationSchedule : schedule.operations)
    hadBindings.push_back(operationSchedule.resourceBindings.has_value());

  for (auto resource : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(resource).value_or(0);
    if (limit == 0)
      continue;
    SmallVector<ResourceUse, 8> uses;
    bool hasStaticBindings = false;
    bool hasMissingBindings = false;
    for (unsigned operationIndex = 0, e = schedule.operations.size();
         operationIndex != e; ++operationIndex) {
      auto resourceIt = llvm::find(limitedResources[operationIndex], resource);
      if (resourceIt == limitedResources[operationIndex].end())
        continue;
      auto &operationSchedule = schedule.operations[operationIndex];
      hasStaticBindings |= hadBindings[operationIndex];
      hasMissingBindings |= !hadBindings[operationIndex];
      uses.push_back(
          {operationSchedule.startTime, operationIndex,
           static_cast<unsigned>(resourceIt -
                                 limitedResources[operationIndex].begin())});
    }
    if (hasStaticBindings && hasMissingBindings)
      return prob.getContainingOp()->emitError()
             << "cannot capture mixed bound and unbound users of resource '"
             << resource.getValue() << "'";
    if (!hasMissingBindings)
      continue;

    for (const auto &use : uses) {
      auto &operationSchedule = schedule.operations[use.operationIndex];
      if (!operationSchedule.resourceBindings)
        operationSchedule.resourceBindings.emplace(
            limitedResources[use.operationIndex].size(), 0);
    }
    llvm::sort(uses, [](const ResourceUse &lhs, const ResourceUse &rhs) {
      return std::tie(lhs.startTime, lhs.operationIndex) <
             std::tie(rhs.startTime, rhs.operationIndex);
    });
    SmallVector<unsigned> nextAvailable(limit, 0);
    unsigned hold = prob.getResourceInitiationInterval(resource).value_or(1);
    for (const auto &use : uses) {
      auto instance = llvm::find_if(nextAvailable, [&](unsigned available) {
        return available <= use.startTime;
      });
      if (instance == nextAvailable.end())
        return prob.getContainingOp()->emitError()
               << "could not synthesize a binding for verified resource '"
               << resource.getValue() << "'";
      unsigned instanceIndex = instance - nextAvailable.begin();
      *instance = use.startTime + hold;
      (*schedule.operations[use.operationIndex]
            .resourceBindings)[use.resourceIndex] = instanceIndex;
    }
  }
  return success();
}

/// Color equal-length circular reservations. Prefer a linear greedy pass, then
/// use bounded DSATUR for small cases where a fixed cut hides a valid static
/// coloring. Static coloring is optional, so larger or exhausted searches can
/// safely fall back to a rotating certificate.
static bool tryColorModuloResource(ArrayRef<ResourceUse> uses, unsigned period,
                                   unsigned hold, unsigned limit,
                                   SmallVectorImpl<unsigned> &colors) {
  colors.assign(uses.size(), 0);
  SmallVector<SmallVector<unsigned, 4>> lanes(limit);
  bool greedySucceeded = true;
  for (auto [useIndex, use] : llvm::enumerate(uses)) {
    unsigned phase = use.startTime % period;
    auto lane = llvm::find_if(lanes, [&](const auto &phases) {
      return phases.empty() || phase >= phases.back() + hold;
    });
    if (lane == lanes.end()) {
      greedySucceeded = false;
      break;
    }
    colors[useIndex] = lane - lanes.begin();
    lane->push_back(phase);
  }
  if (greedySucceeded && llvm::all_of(lanes, [&](const auto &lane) {
        return lane.empty() || lane.front() + period >= lane.back() + hold;
      }))
    return true;

  constexpr unsigned exactNodeLimit = 32;
  constexpr uint64_t searchBudget = 10000;
  if (uses.size() > exactNodeLimit || limit > 64)
    return false;

  SmallVector<SmallVector<unsigned, 8>> conflicts(uses.size());
  for (unsigned lhs = 0, e = uses.size(); lhs != e; ++lhs) {
    unsigned lhsPhase = uses[lhs].startTime % period;
    for (unsigned rhs = lhs + 1; rhs != e; ++rhs) {
      unsigned rhsPhase = uses[rhs].startTime % period;
      unsigned forward = (rhsPhase + period - lhsPhase) % period;
      unsigned reverse = (lhsPhase + period - rhsPhase) % period;
      if (forward >= hold && reverse >= hold)
        continue;
      conflicts[lhs].push_back(rhs);
      conflicts[rhs].push_back(lhs);
    }
  }

  SmallVector<int> assignments(uses.size(), -1);
  uint64_t remainingBudget = searchBudget;
  auto search = [&](auto &&self, unsigned colored) -> bool {
    if (remainingBudget == 0)
      return false;
    --remainingBudget;
    if (colored == uses.size())
      return true;

    unsigned selected = uses.size();
    unsigned selectedSaturation = 0;
    unsigned selectedDegree = 0;
    for (unsigned candidate = 0, e = uses.size(); candidate != e; ++candidate) {
      if (assignments[candidate] >= 0)
        continue;
      uint64_t adjacentColors = 0;
      for (unsigned adjacent : conflicts[candidate])
        if (assignments[adjacent] >= 0)
          adjacentColors |= uint64_t(1) << assignments[adjacent];
      unsigned saturation = llvm::popcount(adjacentColors);
      unsigned degree = conflicts[candidate].size();
      if (selected == uses.size() || saturation > selectedSaturation ||
          (saturation == selectedSaturation && degree > selectedDegree)) {
        selected = candidate;
        selectedSaturation = saturation;
        selectedDegree = degree;
      }
    }

    uint64_t unavailable = 0;
    for (unsigned adjacent : conflicts[selected])
      if (assignments[adjacent] >= 0)
        unavailable |= uint64_t(1) << assignments[adjacent];

    int maximumColor = -1;
    for (int assignment : assignments)
      maximumColor = std::max(maximumColor, assignment);
    unsigned colorEnd =
        std::min(limit, static_cast<unsigned>(maximumColor + 2));
    for (unsigned color = 0; color != colorEnd; ++color) {
      if (unavailable & (uint64_t(1) << color))
        continue;
      assignments[selected] = color;
      if (self(self, colored + 1))
        return true;
      assignments[selected] = -1;
    }
    return false;
  };
  if (!search(search, 0))
    return false;
  for (auto [index, assignment] : llvm::enumerate(assignments))
    colors[index] = assignment;
  return true;
}

/// Try to color a modulo reservation table with one fixed instance per
/// operation. A valid modulo reservation table need not have a static coloring
/// (successive iterations may have to rotate instances), so failure is not an
/// error and the caller can retain explicit rotating reservations instead.
static bool trySynthesizeModuloBindings(
    SharedOperatorsProblem &prob, unsigned period,
    ArrayRef<SmallVector<Problem::ResourceType, 2>> limitedResources,
    ResourceScheduleCertificate &schedule) {
  SmallVector<SmallVector<unsigned, 2>> bindings(schedule.operations.size());
  for (unsigned operationIndex = 0, e = schedule.operations.size();
       operationIndex != e; ++operationIndex)
    bindings[operationIndex].resize(limitedResources[operationIndex].size());

  for (auto resource : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(resource).value_or(0);
    if (limit == 0)
      continue;
    unsigned hold = prob.getResourceInitiationInterval(resource).value_or(1);
    if (hold > period)
      return false;

    SmallVector<ResourceUse, 8> uses;
    for (unsigned operationIndex = 0, e = schedule.operations.size();
         operationIndex != e; ++operationIndex) {
      auto resourceIt = llvm::find(limitedResources[operationIndex], resource);
      if (resourceIt == limitedResources[operationIndex].end())
        continue;
      uses.push_back(
          {schedule.operations[operationIndex].startTime, operationIndex,
           static_cast<unsigned>(resourceIt -
                                 limitedResources[operationIndex].begin())});
    }
    llvm::sort(uses, [period](const ResourceUse &lhs, const ResourceUse &rhs) {
      return std::make_tuple(lhs.startTime % period, lhs.operationIndex) <
             std::make_tuple(rhs.startTime % period, rhs.operationIndex);
    });

    SmallVector<unsigned> colors;
    if (!tryColorModuloResource(uses, period, hold, limit, colors))
      return false;
    for (auto [use, color] : llvm::zip(uses, colors))
      bindings[use.operationIndex][use.resourceIndex] = color;
  }

  for (unsigned operationIndex = 0, e = schedule.operations.size();
       operationIndex != e; ++operationIndex)
    if (!bindings[operationIndex].empty())
      schedule.operations[operationIndex].resourceBindings.emplace(
          std::move(bindings[operationIndex]));
  return true;
}

static FailureOr<ResourceScheduleCertificate>
captureResourceSchedule(SharedOperatorsProblem &prob,
                        std::optional<unsigned> period) {
  DenseMap<Problem::ResourceType, unsigned> limits;
  for (auto resource : prob.getResourceTypes())
    limits[resource] = prob.getLimit(resource).value_or(0);

  ResourceScheduleCertificate schedule;
  SmallVector<SmallVector<Problem::ResourceType, 2>> limitedResources;
  schedule.operations.reserve(prob.getOperations().size());
  limitedResources.reserve(prob.getOperations().size());
  for (auto *op : prob.getOperations()) {
    auto startTime = prob.getStartTime(op);
    if (!startTime)
      return op->emitError(
          "cannot capture a schedule certificate without a start time");

    OperationScheduleCertificate operationSchedule{*startTime};
    if (auto bindings = prob.getResourceBindings(op))
      operationSchedule.resourceBindings.emplace(bindings->begin(),
                                                 bindings->end());
    auto operationResources = getLimitedResources(prob, op, limits);
    schedule.operations.push_back(std::move(operationSchedule));
    limitedResources.push_back(std::move(operationResources));
  }

  if (!period) {
    if (failed(synthesizeAcyclicBindings(prob, limitedResources, schedule)))
      return failure();
    return schedule;
  }

  bool hasBindings = false;
  bool hasUnboundResourceUses = false;
  for (auto [operationSchedule, resources] :
       llvm::zip(schedule.operations, limitedResources)) {
    if (resources.empty())
      continue;
    hasBindings |= operationSchedule.resourceBindings.has_value();
    hasUnboundResourceUses |= !operationSchedule.resourceBindings.has_value();
  }
  if (hasBindings && hasUnboundResourceUses)
    return prob.getContainingOp()->emitError(
        "cannot capture a modulo schedule with mixed bound and unbound "
        "resource users");
  if (hasBindings ||
      trySynthesizeModuloBindings(prob, *period, limitedResources, schedule))
    return schedule;

  for (unsigned operationIndex = 0, e = schedule.operations.size();
       operationIndex != e; ++operationIndex)
    for (auto resource : limitedResources[operationIndex])
      schedule.operations[operationIndex].rotatingReservations.push_back(
          {resource, schedule.operations[operationIndex].startTime % *period,
           *period, prob.getResourceInitiationInterval(resource).value_or(1),
           limits.lookup(resource)});
  return schedule;
}

static LogicalResult validateResourceScheduleCertificate(
    SharedOperatorsProblem &prob, Operation *lastOp,
    const ResourceParetoPoint &point, bool isModulo) {
  if (!prob.hasOperation(lastOp))
    return prob.getContainingOp()->emitError(
        "Pareto point objective operation is not part of the problem");
  if (isModulo ? point.initiationInterval == 0 : point.initiationInterval != 0)
    return prob.getContainingOp()->emitError(
        isModulo ? "a modulo Pareto point requires a non-zero II"
                 : "an acyclic Pareto point must have II zero");
  if (point.schedule.operations.size() != prob.getOperations().size())
    return prob.getContainingOp()->emitError()
           << "schedule certificate contains "
           << point.schedule.operations.size() << " operations, expected "
           << prob.getOperations().size();

  DenseMap<Problem::ResourceType, unsigned> limits;
  if (failed(validateResourceAllocation(prob, point.allocation, limits)))
    return failure();

  // A resource cannot mix static and rotating users because the problem's
  // binding property represents a complete assignment for that resource.
  DenseMap<Problem::ResourceType, unsigned> bindingStyles;
  unsigned operationIndex = 0;
  for (auto *op : prob.getOperations()) {
    const auto &operationSchedule = point.schedule.operations[operationIndex++];
    auto limitedResources = getLimitedResources(prob, op, limits);
    const auto &bindings = operationSchedule.resourceBindings;
    if (bindings) {
      if (limitedResources.empty())
        return op->emitError(
            "schedule certificate binds an operation without limited "
            "resources");
      if (bindings->size() != limitedResources.size())
        return op->emitError(
            "schedule certificate has an invalid number of resource "
            "bindings");
      if (!operationSchedule.rotatingReservations.empty())
        return op->emitError(
            "schedule certificate mixes static and rotating assignments for "
            "one operation");
      for (auto [index, resource] : llvm::enumerate(limitedResources)) {
        if ((*bindings)[index] >= limits.lookup(resource))
          return op->emitError()
                 << "schedule certificate binding for resource '"
                 << resource.getValue() << "' is out of range";
        auto [style, inserted] = bindingStyles.try_emplace(resource, 1);
        if (!inserted && style->second != 1)
          return op->emitError()
                 << "schedule certificate mixes static and rotating users of "
                    "resource '"
                 << resource.getValue() << "'";
      }
      continue;
    }

    if (limitedResources.empty()) {
      if (!operationSchedule.rotatingReservations.empty())
        return op->emitError(
            "schedule certificate reserves a resource not used by the "
            "operation");
      continue;
    }
    if (!isModulo)
      return op->emitError(
          "an acyclic resource schedule certificate requires static "
          "bindings");
    if (operationSchedule.rotatingReservations.size() !=
        limitedResources.size())
      return op->emitError(
          "schedule certificate has an invalid number of rotating resource "
          "reservations");

    for (auto [index, resource] : llvm::enumerate(limitedResources)) {
      const auto &reservation = operationSchedule.rotatingReservations[index];
      unsigned hold = prob.getResourceInitiationInterval(resource).value_or(1);
      if (reservation.resource != resource ||
          reservation.phase !=
              operationSchedule.startTime % point.initiationInterval ||
          reservation.period != point.initiationInterval ||
          reservation.hold != hold ||
          reservation.instances != limits.lookup(resource))
        return op->emitError()
               << "rotating reservation for resource '" << resource.getValue()
               << "' does not match the problem or Pareto point";
      auto [style, inserted] = bindingStyles.try_emplace(resource, 2);
      if (!inserted && style->second != 2)
        return op->emitError()
               << "schedule certificate mixes static and rotating users of "
                  "resource '"
               << resource.getValue() << "'";
    }
  }

  operationIndex = 0;
  for (auto *op : prob.getOperations()) {
    if (op == lastOp &&
        point.schedule.operations[operationIndex].startTime != point.latency)
      return op->emitError(
          "schedule certificate objective start time does not match its "
          "recorded latency");
    ++operationIndex;
  }
  return success();
}

static void applyResourceScheduleCertificate(SharedOperatorsProblem &prob,
                                             const ResourceParetoPoint &point) {
  applyResourceAllocation(prob, point.allocation);
  prob.clearResourceBindings();
  for (auto [op, operationSchedule] :
       llvm::zip(prob.getOperations(), point.schedule.operations)) {
    prob.setStartTime(op, operationSchedule.startTime);
    if (operationSchedule.resourceBindings)
      prob.setResourceBindings(
          op, SmallVector<unsigned>(operationSchedule.resourceBindings->begin(),
                                    operationSchedule.resourceBindings->end()));
  }
}

template <typename ProblemT>
static FailureOr<SmallVector<ResourceParetoPoint>> exploreResourceParetoImpl(
    ProblemT &prob, Operation *lastOp, ArrayRef<ResourceAllocation> allocations,
    ResourceCostFunction costFunction,
    detail::ResourceScheduleFunction schedule, StringRef schedulerName,
    function_ref<unsigned()> getInitiationInterval) {
  if (allocations.empty())
    return prob.getContainingOp()->emitError()
           << schedulerName
           << " Pareto exploration requires at least one allocation";

  SmallVector<ResourceParetoPoint> evaluated;
  evaluated.reserve(allocations.size());
  for (const auto &allocation : allocations) {
    DenseMap<Problem::ResourceType, unsigned> limits;
    if (failed(validateResourceAllocation(prob, allocation, limits)))
      return failure();
    applyResourceAllocation(prob, allocation);

    if (failed(schedule()) || failed(prob.verify()))
      return failure();
    unsigned initiationInterval = getInitiationInterval();
    auto certificate = captureResourceSchedule(
        prob, initiationInterval ? std::optional<unsigned>(initiationInterval)
                                 : std::nullopt);
    if (failed(certificate))
      return failure();
    evaluated.push_back({allocation, *prob.getStartTime(lastOp),
                         costFunction(allocation), initiationInterval,
                         std::move(*certificate)});
  }

  SmallVector<ResourceParetoPoint> frontier;
  for (unsigned i = 0, e = evaluated.size(); i != e; ++i) {
    bool dominated = false;
    for (unsigned j = 0; j != e; ++j) {
      if (i == j)
        continue;
      const auto &candidate = evaluated[i];
      const auto &other = evaluated[j];
      if (other.resourceCost <= candidate.resourceCost &&
          other.latency <= candidate.latency &&
          other.initiationInterval <= candidate.initiationInterval &&
          (other.resourceCost < candidate.resourceCost ||
           other.latency < candidate.latency ||
           other.initiationInterval < candidate.initiationInterval)) {
        dominated = true;
        break;
      }
    }
    if (!dominated)
      frontier.push_back(evaluated[i]);
  }

  auto selected =
      llvm::min_element(frontier, [](const ResourceParetoPoint &lhs,
                                     const ResourceParetoPoint &rhs) {
        return std::tie(lhs.resourceCost, lhs.initiationInterval, lhs.latency) <
               std::tie(rhs.resourceCost, rhs.initiationInterval, rhs.latency);
      });
  if (failed(applyResourceParetoPoint(prob, lastOp, *selected)))
    return failure();
  return frontier;
}

} // namespace

LogicalResult
scheduling::applyResourceParetoPoint(SharedOperatorsProblem &prob,
                                     Operation *lastOp,
                                     const ResourceParetoPoint &point) {
  if (failed(validateResourceScheduleCertificate(prob, lastOp, point, false)))
    return failure();
  applyResourceScheduleCertificate(prob, point);
  return failed(prob.check()) || failed(prob.verify()) ? failure() : success();
}

LogicalResult
scheduling::applyResourceParetoPoint(ModuloProblem &prob, Operation *lastOp,
                                     const ResourceParetoPoint &point) {
  if (failed(validateResourceScheduleCertificate(prob, lastOp, point, true)))
    return failure();
  prob.setInitiationInterval(point.initiationInterval);
  applyResourceScheduleCertificate(prob, point);
  return failed(prob.check()) || failed(prob.verify()) ? failure() : success();
}

FailureOr<SmallVector<ResourceParetoPoint>>
scheduling::detail::exploreResourcePareto(
    SharedOperatorsProblem &prob, Operation *lastOp,
    ArrayRef<ResourceAllocation> allocations, ResourceCostFunction costFunction,
    ResourceScheduleFunction schedule, StringRef schedulerName) {
  return exploreResourceParetoImpl(prob, lastOp, allocations, costFunction,
                                   schedule, schedulerName, [] { return 0u; });
}

FailureOr<SmallVector<ResourceParetoPoint>>
scheduling::detail::exploreResourcePareto(
    ModuloProblem &prob, Operation *lastOp,
    ArrayRef<ResourceAllocation> allocations, ResourceCostFunction costFunction,
    ResourceScheduleFunction schedule, StringRef schedulerName) {
  return exploreResourceParetoImpl(
      prob, lastOp, allocations, costFunction, schedule, schedulerName,
      [&] { return *prob.getInitiationInterval(); });
}
