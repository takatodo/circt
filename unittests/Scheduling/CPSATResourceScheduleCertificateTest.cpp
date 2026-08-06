//===- CPSATResourceScheduleCertificateTest.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "gtest/gtest.h"


using namespace mlir;
using namespace circt;
using namespace circt::scheduling;

namespace {

struct TestIR {
  TestIR()
      : module(ModuleOp::create(UnknownLoc::get(&context))), builder(&context) {
    builder.setInsertionPointToStart(module->getBody());
  }

  Operation *createOperation() {
    OperationState state(module->getLoc(), "test.operation");
    return builder.create(state);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  OpBuilder builder;
};

TEST(CPSATResourceScheduleCertificateTest, ReappliesEveryAcyclicParetoPoint) {
  TestIR ir;
  auto *lhs = ir.createOperation();
  auto *rhs = ir.createOperation();
  auto *last = ir.createOperation();

  SharedOperatorsProblem problem(*ir.module);
  auto compute = problem.getOrInsertOperatorType("compute");
  auto sink = problem.getOrInsertOperatorType("sink");
  problem.setLatency(compute, 1);
  problem.setLatency(sink, 1);
  auto resource = problem.getOrInsertResourceType("compute-resource");
  for (auto *op : {lhs, rhs}) {
    problem.insertOperation(op);
    problem.setLinkedOperatorType(op, compute);
    problem.setLinkedResourceTypes(op, {resource});
  }
  problem.insertOperation(last);
  problem.setLinkedOperatorType(last, sink);
  ASSERT_TRUE(succeeded(problem.insertDependence({lhs, last})));
  ASSERT_TRUE(succeeded(problem.insertDependence({rhs, last})));
  ASSERT_TRUE(succeeded(problem.check()));

  SmallVector<ResourceAllocation> allocations = {
      {{ResourceLimit{resource, 1}}}, {{ResourceLimit{resource, 2}}}};
  CPSATSchedulerOptions options;
  options.numWorkers = 1;
  auto frontier = exploreCPSATPareto(problem, last, allocations, options);
  ASSERT_TRUE(succeeded(frontier));
  ASSERT_EQ(frontier->size(), 2u);

  for (const auto &point : *frontier) {
    ASSERT_EQ(point.schedule.operations.size(), 3u);
    for (unsigned operationIndex : {0u, 1u}) {
      const auto &operation = point.schedule.operations[operationIndex];
      ASSERT_TRUE(operation.resourceBindings);
      ASSERT_EQ(operation.resourceBindings->size(), 1u);
      EXPECT_LT(operation.resourceBindings->front(),
                point.allocation.limits.front().limit);
      EXPECT_TRUE(operation.rotatingReservations.empty());
    }
    ASSERT_TRUE(succeeded(applyResourceParetoPoint(problem, last, point)));
    EXPECT_EQ(*problem.getStartTime(last), point.latency);
    EXPECT_TRUE(succeeded(problem.verify()));
  }

  auto parallelPoint = llvm::find_if(*frontier, [](const auto &point) {
    return point.allocation.limits.front().limit == 2;
  });
  ASSERT_NE(parallelPoint, frontier->end());
  EXPECT_EQ(parallelPoint->latency, 1u);
  EXPECT_NE(parallelPoint->schedule.operations[0].resourceBindings->front(),
            parallelPoint->schedule.operations[1].resourceBindings->front());
}

TEST(CPSATResourceScheduleCertificateTest, ReappliesEveryModuloParetoPoint) {
  TestIR ir;
  auto *recurrenceOp = ir.createOperation();
  auto *last = ir.createOperation();

  ModuloProblem problem(*ir.module);
  auto compute = problem.getOrInsertOperatorType("compute");
  auto sink = problem.getOrInsertOperatorType("sink");
  problem.setLatency(compute, 1);
  problem.setLatency(sink, 1);
  auto resource = problem.getOrInsertResourceType("compute-resource");
  problem.setResourceInitiationInterval(resource, 3);
  problem.insertOperation(recurrenceOp);
  problem.setLinkedOperatorType(recurrenceOp, compute);
  problem.setLinkedResourceTypes(recurrenceOp, {resource});
  problem.insertOperation(last);
  problem.setLinkedOperatorType(last, sink);

  Problem::Dependence recurrence(recurrenceOp, recurrenceOp);
  ASSERT_TRUE(succeeded(problem.insertDependence(recurrence)));
  problem.setDistance(recurrence, 1);
  ASSERT_TRUE(succeeded(problem.insertDependence({recurrenceOp, last})));
  ASSERT_TRUE(succeeded(problem.check()));

  SmallVector<ResourceAllocation> allocations = {
      {{ResourceLimit{resource, 1}}}, {{ResourceLimit{resource, 2}}}};
  CPSATSchedulerOptions options;
  options.numWorkers = 1;
  auto frontier = exploreCPSATPareto(problem, last, allocations, options);
  ASSERT_TRUE(succeeded(frontier));
  ASSERT_EQ(frontier->size(), 2u);

  for (const auto &point : *frontier) {
    ASSERT_EQ(point.schedule.operations.size(), 2u);
    ASSERT_TRUE(succeeded(applyResourceParetoPoint(problem, last, point)));
    EXPECT_EQ(*problem.getInitiationInterval(), point.initiationInterval);
    EXPECT_EQ(*problem.getStartTime(last), point.latency);
    EXPECT_TRUE(succeeded(problem.verify()));
  }

  auto oneInstance = llvm::find_if(*frontier, [](const auto &point) {
    return point.allocation.limits.front().limit == 1;
  });
  auto twoInstances = llvm::find_if(*frontier, [](const auto &point) {
    return point.allocation.limits.front().limit == 2;
  });
  ASSERT_NE(oneInstance, frontier->end());
  ASSERT_NE(twoInstances, frontier->end());
  EXPECT_EQ(oneInstance->initiationInterval, 3u);
  EXPECT_EQ(twoInstances->initiationInterval, 2u);

  const auto &staticOperation = oneInstance->schedule.operations.front();
  ASSERT_TRUE(staticOperation.resourceBindings);
  ASSERT_EQ(staticOperation.resourceBindings->size(), 1u);
  EXPECT_EQ(staticOperation.resourceBindings->front(), 0u);
  EXPECT_TRUE(staticOperation.rotatingReservations.empty());

  const auto &rotatingOperation = twoInstances->schedule.operations.front();
  EXPECT_FALSE(rotatingOperation.resourceBindings);
  ASSERT_EQ(rotatingOperation.rotatingReservations.size(), 1u);
  const auto &reservation = rotatingOperation.rotatingReservations.front();
  EXPECT_TRUE(reservation.resource == resource);
  EXPECT_EQ(reservation.phase,
            rotatingOperation.startTime % twoInstances->initiationInterval);
  EXPECT_EQ(reservation.period, twoInstances->initiationInterval);
  EXPECT_EQ(reservation.hold, 3u);
  EXPECT_EQ(reservation.instances, 2u);
}


} // namespace
