//===- Problems.cpp - Modeling of scheduling problems ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements base classes for scheduling problems.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/DependenceIterator.h"

#include "mlir/IR/Operation.h"

#include <algorithm>
#include <tuple>

using namespace circt;
using namespace circt::scheduling;
using namespace circt::scheduling::detail;

//===----------------------------------------------------------------------===//
// Problem
//===----------------------------------------------------------------------===//

LogicalResult Problem::insertDependence(Dependence dep) {
  Operation *src = dep.getSource();
  Operation *dst = dep.getDestination();

  // Fail early on invalid dependences (src == dst == null), and def-use
  // dependences that cannot be added because the source value is not the result
  // of an operation (e.g., a BlockArgument).
  if (!src || !dst)
    return failure();

  // record auxiliary dependences explicitly
  if (dep.isAuxiliary())
    auxDependences[dst].insert(src);

  // auto-register the endpoints
  operations.insert(src);
  operations.insert(dst);

  return success();
}

Problem::OperatorType Problem::getOrInsertOperatorType(StringRef name) {
  auto opr = OperatorType::get(containingOp->getContext(), name);
  operatorTypes.insert(opr);
  return opr;
}

Problem::ResourceType Problem::getOrInsertResourceType(StringRef name) {
  auto rsrc = ResourceType::get(containingOp->getContext(), name);
  resourceTypes.insert(rsrc);
  return rsrc;
}

Problem::DependenceRange Problem::getDependences(Operation *op) {
  return DependenceRange(DependenceIterator(*this, op),
                         DependenceIterator(*this, op, /*end=*/true));
}

Problem::PropertyStringVector Problem::getProperties(Operation *op) {
  PropertyStringVector psv;
  if (auto linkedOpr = getLinkedOperatorType(op))
    psv.emplace_back("linkedOpr", (*linkedOpr).str());
  if (auto startTime = getStartTime(op))
    psv.emplace_back("startTime", std::to_string(*startTime));
  return psv;
}

Problem::PropertyStringVector Problem::getProperties(Dependence dep) {
  return {};
}

Problem::PropertyStringVector Problem::getProperties(OperatorType opr) {
  PropertyStringVector psv;
  if (auto latency = getLatency(opr))
    psv.emplace_back("latency", std::to_string(*latency));
  return psv;
}

Problem::PropertyStringVector Problem::getProperties() { return {}; }

Problem::PropertyStringVector Problem::getProperties(ResourceType rsrc) {
  return {};
}

LogicalResult Problem::checkLinkedOperatorType(Operation *op) {
  if (!getLinkedOperatorType(op))
    return op->emitError("Operation is not linked to an operator type");
  if (!hasOperatorType(*getLinkedOperatorType(op)))
    return op->emitError("Operation uses an unregistered operator type");
  return success();
}

LogicalResult Problem::checkLatency(Operation *op) {
  auto maybeOpr = getLinkedOperatorType(op);
  if (!maybeOpr)
    return getContainingOp()->emitError()
           << "Operation is missing a linked operator type";

  if (!getLatency(*maybeOpr))
    return getContainingOp()->emitError()
           << "Operator type '" << maybeOpr->getValue() << "' has no latency";

  return success();
}

LogicalResult Problem::check() {
  for (auto *op : getOperations()) {
    if (failed(checkLinkedOperatorType(op)))
      return failure();

    if (failed(checkLatency(op)))
      return failure();
  }

  return success();
}

LogicalResult Problem::verifyStartTime(Operation *op) {
  if (!getStartTime(op))
    return op->emitError("Operation has no start time");
  return success();
}

LogicalResult Problem::verifyPrecedence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);

  // check if i's result is available before j starts
  if (!(stI + latI <= stJ))
    return getContainingOp()->emitError()
           << "Precedence violated for dependence." << "\n  from: " << *i
           << ", result available in t=" << (stI + latI) << "\n  to:   " << *j
           << ", starts in t=" << stJ;

  return success();
}

LogicalResult Problem::verify() {
  for (auto *op : getOperations())
    if (failed(verifyStartTime(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(verifyPrecedence(dep)))
        return failure();

  return success();
}

std::optional<unsigned> Problem::getEndTime(Operation *op) {
  if (auto startTime = getStartTime(op))
    if (auto opType = getLinkedOperatorType(op))
      if (auto latency = getLatency(*opType))
        return startTime.value() + latency.value();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// CyclicProblem
//===----------------------------------------------------------------------===//

Problem::PropertyStringVector CyclicProblem::getProperties(Dependence dep) {
  auto psv = Problem::getProperties(dep);
  if (auto distance = getDistance(dep))
    psv.emplace_back("distance", std::to_string(*distance));
  return psv;
}

Problem::PropertyStringVector CyclicProblem::getProperties() {
  auto psv = Problem::getProperties();
  if (auto ii = getInitiationInterval())
    psv.emplace_back("II", std::to_string(*ii));
  return psv;
}

LogicalResult CyclicProblem::verifyPrecedence(Dependence dep) {
  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);
  unsigned dist = getDistance(dep).value_or(0); // optional property
  unsigned ii = *getInitiationInterval();

  // check if i's result is available before j starts (dist iterations later)
  if (!(stI + latI <= stJ + dist * ii))
    return getContainingOp()->emitError()
           << "Precedence violated for dependence." << "\n  from: " << *i
           << ", result available in t=" << (stI + latI) << "\n  to:   " << *j
           << ", starts in t=" << stJ << "\n  dist: " << dist << ", II=" << ii;

  return success();
}

LogicalResult CyclicProblem::verifyInitiationInterval() {
  if (!getInitiationInterval() || *getInitiationInterval() == 0)
    return getContainingOp()->emitError("Invalid initiation interval");
  return success();
}

LogicalResult CyclicProblem::verify() {
  if (failed(verifyInitiationInterval()) || failed(Problem::verify()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ChainingProblem
//===----------------------------------------------------------------------===//

Problem::PropertyStringVector ChainingProblem::getProperties(Operation *op) {
  auto psv = Problem::getProperties(op);
  if (auto stic = getStartTimeInCycle(op))
    psv.emplace_back("start time in cycle", std::to_string(*stic));
  return psv;
}

Problem::PropertyStringVector ChainingProblem::getProperties(OperatorType opr) {
  auto psv = Problem::getProperties(opr);
  if (auto incDelay = getIncomingDelay(opr))
    psv.emplace_back("incoming delay", std::to_string(*incDelay));
  if (auto outDelay = getOutgoingDelay(opr))
    psv.emplace_back("outgoing delay", std::to_string(*outDelay));
  return psv;
}

LogicalResult ChainingProblem::checkDelays(OperatorType opr) {
  auto incomingDelay = getIncomingDelay(opr);
  auto outgoingDelay = getOutgoingDelay(opr);

  if (!incomingDelay || !outgoingDelay)
    return getContainingOp()->emitError()
           << "Missing delays for operator type '" << opr.getAttr() << "'";

  float iDel = *incomingDelay;
  float oDel = *outgoingDelay;

  if (iDel < 0.0f || oDel < 0.0f)
    return getContainingOp()->emitError()
           << "Negative delays for operator type '" << opr.getAttr() << "'";

  if (*getLatency(opr) == 0 && iDel != oDel)
    return getContainingOp()->emitError()
           << "Incoming & outgoing delay must be equal for zero-latency "
              "operator type '"
           << opr.getAttr() << "'";

  return success();
}

LogicalResult ChainingProblem::verifyStartTimeInCycle(Operation *op) {
  auto startTimeInCycle = getStartTimeInCycle(op);
  if (!startTimeInCycle || *startTimeInCycle < 0.0f)
    return op->emitError("Operation has no non-negative start time in cycle");
  return success();
}

LogicalResult ChainingProblem::verifyPrecedenceInCycle(Dependence dep) {
  // Auxiliary edges don't transport values.
  if (dep.isAuxiliary())
    return success();

  Operation *i = dep.getSource();
  Operation *j = dep.getDestination();

  unsigned stI = *getStartTime(i);
  unsigned latI = *getLatency(*getLinkedOperatorType(i));
  unsigned stJ = *getStartTime(j);

  // If `i` finishes a full time step earlier than `j`, its value is registered
  // and thereby available at physical time 0.0 in `j`'s start cycle.
  if (stI + latI < stJ)
    return success();

  // We have stI + latI == stJ, i.e. `i` ends in the same cycle as `j` starts.
  // If `i` is combinational, both ops also start in the same cycle, and we must
  // include `i`'s start time in that cycle in the path delay. Otherwise, `i`
  // started in an earlier cycle and just contributes its outgoing delay to the
  // path.
  float sticI = latI == 0 ? *getStartTimeInCycle(i) : 0.0f;
  float oDelI = *getOutgoingDelay(*getLinkedOperatorType(i));
  float sticJ = *getStartTimeInCycle(j);

  if (!(sticI + oDelI <= sticJ))
    return getContainingOp()->emitError()
           << "Precedence violated in cycle " << stJ
           << " for dependence:" << "\n  from: " << *i
           << ", result after z=" << (sticI + oDelI) << "\n  to:   " << *j
           << ", starts in z=" << sticJ;

  return success();
}

LogicalResult ChainingProblem::check() {
  if (failed(Problem::check()))
    return failure();

  for (auto opr : getOperatorTypes())
    if (failed(checkDelays(opr)))
      return failure();

  return success();
}

LogicalResult ChainingProblem::verify() {
  if (failed(Problem::verify()))
    return failure();

  for (auto *op : getOperations())
    if (failed(verifyStartTimeInCycle(op)))
      return failure();

  for (auto *op : getOperations())
    for (auto dep : getDependences(op))
      if (failed(verifyPrecedenceInCycle(dep)))
        return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// SharedOperatorsProblem
//===----------------------------------------------------------------------===//

Problem::PropertyStringVector
SharedOperatorsProblem::getProperties(ResourceType rsrc) {
  auto psv = Problem::getProperties(rsrc);
  if (auto limit = getLimit(rsrc))
    psv.emplace_back("limit", std::to_string(*limit));
  if (auto ii = getResourceInitiationInterval(rsrc))
    psv.emplace_back("ii", std::to_string(*ii));
  if (auto cost = getResourceCost(rsrc))
    psv.emplace_back("cost", std::to_string(*cost));
  return psv;
}

LogicalResult SharedOperatorsProblem::check() {
  if (failed(Problem::check()))
    return failure();

  for (auto rsrc : getResourceTypes()) {
    if (auto ii = getResourceInitiationInterval(rsrc); ii && *ii == 0)
      return getContainingOp()->emitError()
             << "Resource type '" << rsrc.getValue()
             << "' has an invalid zero initiation interval";
  }
  return success();
}

LogicalResult SharedOperatorsProblem::checkLatency(Operation *op) {
  if (failed(Problem::checkLatency(op)))
    return failure();

  auto maybeRsrcs = getLinkedResourceTypes(op);
  if (!maybeRsrcs)
    return success();

  // `linkedOprType` is not null since it must have been checked by the base
  // class' `checkLatency`.
  OperatorType linkedOprType = *getLinkedOperatorType(op);

  for (auto rsrc : *maybeRsrcs) {
    auto limit = getLimit(rsrc);
    if (limit && *limit > 0 && *getLatency(linkedOprType) == 0)
      return getContainingOp()->emitError()
             << "Operator type '" << linkedOprType.getValue()
             << "' using limited resource '" << rsrc.getValue()
             << "' has zero latency.";
  }
  return success();
}

LogicalResult SharedOperatorsProblem::verifyUtilization(ResourceType rsrc) {
  auto limit = getLimit(rsrc);
  if (!limit || *limit == 0)
    return success();

  llvm::SmallDenseMap<unsigned, unsigned> nOpsPerTimeStep;
  for (auto *op : getOperations()) {
    auto maybeRsrcs = getLinkedResourceTypes(op);
    if (!maybeRsrcs)
      continue;

    if (llvm::none_of(*maybeRsrcs, [&](ResourceType linkedRsrc) {
          return linkedRsrc == rsrc;
        }))
      continue;

    unsigned ii = getResourceInitiationInterval(rsrc).value_or(1);
    for (unsigned time = *getStartTime(op), end = time + ii; time != end;
         ++time)
      ++nOpsPerTimeStep[time];
  }

  for (auto &kv : nOpsPerTimeStep)
    if (kv.second > *limit)
      return getContainingOp()->emitError()
             << "Resource type '" << rsrc.getValue() << "' is oversubscribed."
             << "\n  time step: " << kv.first
             << "\n  #operations: " << kv.second << "\n  limit: " << *limit;

  return success();
}

LogicalResult SharedOperatorsProblem::verifyBindings(ResourceType rsrc) {
  auto limit = getLimit(rsrc);
  if (!limit || *limit == 0)
    return success();

  struct BoundOperation {
    unsigned startTime;
    unsigned binding;
  };
  SmallVector<BoundOperation> boundOperations;
  bool hasAnyBindings = false;
  bool hasMissingBindings = false;
  for (auto *op : getOperations()) {
    auto resources = getLinkedResourceTypes(op);
    if (!resources)
      continue;
    SmallVector<ResourceType> limitedResources;
    for (auto resource : *resources)
      if (getLimit(resource).value_or(0) > 0 &&
          std::find(limitedResources.begin(), limitedResources.end(),
                    resource) == limitedResources.end())
        limitedResources.push_back(resource);

    auto bindings = getResourceBindings(op);
    if (!limitedResources.empty()) {
      hasAnyBindings |= bindings.has_value();
      hasMissingBindings |= !bindings.has_value();
    }
    if (!bindings)
      continue;
    if (bindings->size() != limitedResources.size())
      return op->emitError(
          "Operation has an invalid number of resource bindings");
    for (auto [index, resource] : llvm::enumerate(limitedResources)) {
      if (resource != rsrc)
        continue;
      if ((*bindings)[index] >= *limit)
        return op->emitError() << "Operation binding for resource '"
                               << rsrc.getValue() << "' is out of range";
      boundOperations.push_back({*getStartTime(op), (*bindings)[index]});
    }
  }
  if (!hasAnyBindings)
    return success();
  if (hasMissingBindings)
    return getContainingOp()->emitError(
        "A resource-bound schedule must bind every limited operation");

  llvm::sort(boundOperations,
             [](const BoundOperation &lhs, const BoundOperation &rhs) {
               return std::tie(lhs.binding, lhs.startTime) <
                      std::tie(rhs.binding, rhs.startTime);
             });
  unsigned ii = getResourceInitiationInterval(rsrc).value_or(1);
  for (unsigned index = 1, e = boundOperations.size(); index != e; ++index) {
    const auto &previous = boundOperations[index - 1];
    const auto &current = boundOperations[index];
    if (previous.binding == current.binding &&
        previous.startTime + ii > current.startTime)
      return getContainingOp()->emitError()
             << "Resource instance " << previous.binding << " of type '"
             << rsrc.getValue() << "' has overlapping bindings";
  }
  return success();
}

LogicalResult SharedOperatorsProblem::verify() {
  if (failed(Problem::verify()))
    return failure();

  for (auto rsrc : getResourceTypes())
    if (failed(verifyUtilization(rsrc)))
      return failure();

  for (auto rsrc : getResourceTypes())
    if (failed(verifyBindings(rsrc)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ModuloProblem
//===----------------------------------------------------------------------===//

LogicalResult ModuloProblem::check() {
  return failed(Problem::check()) || failed(SharedOperatorsProblem::check())
             ? failure()
             : success();
}

LogicalResult ModuloProblem::verifyUtilization(ResourceType rsrc) {
  auto limit = getLimit(rsrc);
  if (!limit || *limit == 0)
    return success();

  unsigned ii = *getInitiationInterval();
  llvm::SmallDenseMap<unsigned, unsigned> nOpsPerCongruenceClass;
  for (auto *op : getOperations()) {
    auto maybeRsrcs = getLinkedResourceTypes(op);
    if (!maybeRsrcs)
      continue;

    if (llvm::none_of(*maybeRsrcs, [&](ResourceType linkedRsrc) {
          return linkedRsrc == rsrc;
        }))
      continue;

    unsigned resourceII = getResourceInitiationInterval(rsrc).value_or(1);
    for (unsigned offset = 0; offset != resourceII; ++offset)
      ++nOpsPerCongruenceClass[(*getStartTime(op) + offset) % ii];
  }

  for (auto &kv : nOpsPerCongruenceClass)
    if (kv.second > *limit)
      return getContainingOp()->emitError()
             << "Resource type '" << rsrc.getValue() << "' is oversubscribed."
             << "\n  congruence class: " << kv.first
             << "\n  #reservations: " << kv.second << "\n  limit: " << *limit;

  return success();
}

LogicalResult ModuloProblem::verifyBindings(ResourceType rsrc) {
  auto limit = getLimit(rsrc);
  if (!limit || *limit == 0)
    return success();

  struct BoundOperation {
    unsigned binding;
    unsigned startTime;
  };
  SmallVector<BoundOperation> boundOperations;
  bool hasAnyBindings = false;
  bool hasMissingBindings = false;
  for (auto *op : getOperations()) {
    auto resources = getLinkedResourceTypes(op);
    if (!resources)
      continue;
    SmallVector<ResourceType> limitedResources;
    for (auto resource : *resources)
      if (getLimit(resource).value_or(0) > 0 &&
          !llvm::is_contained(limitedResources, resource))
        limitedResources.push_back(resource);
    auto it = llvm::find(limitedResources, rsrc);
    if (it == limitedResources.end())
      continue;

    auto bindings = getResourceBindings(op);
    if (!bindings || bindings->size() != limitedResources.size()) {
      hasMissingBindings = true;
      continue;
    }
    hasAnyBindings = true;
    unsigned binding = (*bindings)[it - limitedResources.begin()];
    if (binding >= *limit)
      return getContainingOp()->emitError()
             << "Resource binding " << binding << " is out of range for type '"
             << rsrc.getValue() << "'";
    boundOperations.push_back({binding, *getStartTime(op)});
  }
  if (!hasAnyBindings)
    return success();
  if (hasMissingBindings)
    return getContainingOp()->emitError(
        "A resource-bound schedule must bind every limited operation");

  unsigned pipelineII = *getInitiationInterval();
  unsigned resourceII = getResourceInitiationInterval(rsrc).value_or(1);
  if (resourceII > pipelineII)
    return getContainingOp()->emitError()
           << "Static modulo resource bindings require resource initiation "
              "interval no larger than the pipeline II";

  llvm::sort(boundOperations, [pipelineII](const BoundOperation &lhs,
                                           const BoundOperation &rhs) {
    return std::make_tuple(lhs.binding, lhs.startTime % pipelineII) <
           std::make_tuple(rhs.binding, rhs.startTime % pipelineII);
  });
  for (unsigned begin = 0, e = boundOperations.size(); begin != e;) {
    unsigned end = begin + 1;
    while (end != e &&
           boundOperations[end].binding == boundOperations[begin].binding)
      ++end;
    if (end - begin == 1) {
      begin = end;
      continue;
    }
    for (unsigned index = begin; index != end; ++index) {
      const auto &current = boundOperations[index];
      const auto &next = boundOperations[index + 1 == end ? begin : index + 1];
      unsigned currentPhase = current.startTime % pipelineII;
      unsigned nextPhase = next.startTime % pipelineII;
      unsigned distance = nextPhase >= currentPhase
                              ? nextPhase - currentPhase
                              : nextPhase + pipelineII - currentPhase;
      if (distance < resourceII)
        return getContainingOp()->emitError()
               << "Resource instance " << current.binding << " of type '"
               << rsrc.getValue() << " has overlapping modulo bindings";
    }
    begin = end;
  }
  return success();
}

LogicalResult ModuloProblem::verify() {
  if (failed(CyclicProblem::verify()))
    return failure();

  // Don't call SharedOperatorsProblem::verify() here to prevent redundant
  // verification of the base problem.
  for (auto rsrc : getResourceTypes())
    if (failed(verifyUtilization(rsrc)))
      return failure();

  for (auto rsrc : getResourceTypes())
    if (failed(verifyBindings(rsrc)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ChainingCyclicProblem
//===----------------------------------------------------------------------===//

LogicalResult ChainingCyclicProblem::checkDefUse(Dependence dep) {
  if (!dep.isAuxiliary() && (getDistance(dep).value_or(0) != 0))
    return getContainingOp()->emitError()
           << "Def-use dependence cannot have non-zero distance.\n"
           << "On operation: " << *dep.getDestination() << ".\n";
  return success();
}

LogicalResult ChainingCyclicProblem::check() {
  for (auto *op : getOperations())
    for (auto &dep : getDependences(op))
      if (failed(checkDefUse(dep)))
        return failure();

  if (ChainingProblem::check().succeeded() &&
      CyclicProblem::check().succeeded())
    return success();
  return failure();
}

LogicalResult ChainingCyclicProblem::verify() {
  if (ChainingProblem::verify().succeeded() &&
      CyclicProblem::verify().succeeded())
    return success();
  return failure();
}

//===----------------------------------------------------------------------===//
// Dependence
//===----------------------------------------------------------------------===//

Operation *Dependence::getSource() const {
  return isDefUse() ? defUse->get().getDefiningOp() : auxSrc;
}

Operation *Dependence::getDestination() const {
  return isDefUse() ? defUse->getOwner() : auxDst;
}

std::optional<unsigned> Dependence::getSourceIndex() const {
  if (!isDefUse())
    return std::nullopt;

  assert(isa<OpResult>(defUse->get()) && "source is not an operation");
  return dyn_cast<OpResult>(defUse->get()).getResultNumber();
}

std::optional<unsigned> Dependence::getDestinationIndex() const {
  if (!isDefUse())
    return std::nullopt;
  return defUse->getOperandNumber();
}

Dependence::TupleRepr Dependence::getAsTuple() const {
  return TupleRepr(getSource(), getDestination(), getSourceIndex(),
                   getDestinationIndex());
}

bool Dependence::operator==(const Dependence &other) const {
  return getAsTuple() == other.getAsTuple();
}

//===----------------------------------------------------------------------===//
// DependenceIterator
//===----------------------------------------------------------------------===//

DependenceIterator::DependenceIterator(Problem &problem, Operation *op,
                                       bool end)
    : problem(problem), op(op), operandIdx(0), auxPredIdx(0), auxPreds(nullptr),
      dep() {
  if (!end) {
    if (problem.auxDependences.count(op))
      auxPreds = &problem.auxDependences[op];

    findNextDependence();
  }
}

void DependenceIterator::findNextDependence() {
  // Yield dependences corresponding to values used by `op`'s operands...
  while (operandIdx < op->getNumOperands()) {
    dep = Dependence(&op->getOpOperand(operandIdx++));
    Operation *src = dep.getSource();

    // ... but only if they are outgoing from operations that are registered in
    // the scheduling problem.
    if (src && problem.hasOperation(src))
      return;
  }

  // Then, yield auxiliary dependences, if present.
  if (auxPreds && auxPredIdx < auxPreds->size()) {
    dep = Dependence((*auxPreds)[auxPredIdx++], op);
    return;
  }

  // An invalid dependence signals the end of iteration.
  dep = Dependence();
}
