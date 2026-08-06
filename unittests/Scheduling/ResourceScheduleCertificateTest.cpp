//===- ResourceScheduleCertificateTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

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

TEST(ResourceScheduleCertificateTest, ReappliesAcyclicParetoPoint) {
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
  NodeRLSchedulerOptions options;
  options.episodes = 8;
  options.seed = 7;
  auto frontier = exploreNodeRLPareto(problem, last, allocations, options);
  ASSERT_TRUE(succeeded(frontier));
  ASSERT_EQ(frontier->size(), 2u);
  EXPECT_EQ(*problem.getLimit(resource), 1u);
  for (const auto &point : *frontier) {
    ASSERT_TRUE(succeeded(applyResourceParetoPoint(problem, last, point)));
    EXPECT_EQ(*problem.getStartTime(last), point.latency);
  }

  auto parallelPoint = llvm::find_if(*frontier, [](const auto &point) {
    return point.allocation.limits.front().limit == 2;
  });
  ASSERT_NE(parallelPoint, frontier->end());
  ASSERT_EQ(parallelPoint->schedule.operations.size(), 3u);
  for (unsigned index : {0u, 1u}) {
    const auto &operation = parallelPoint->schedule.operations[index];
    ASSERT_TRUE(operation.resourceBindings);
    EXPECT_EQ(operation.resourceBindings->size(), 1u);
    EXPECT_TRUE(operation.rotatingReservations.empty());
  }

  ASSERT_TRUE(
      succeeded(applyResourceParetoPoint(problem, last, *parallelPoint)));
  EXPECT_EQ(*problem.getLimit(resource), 2u);
  EXPECT_EQ(*problem.getStartTime(last), parallelPoint->latency);
  EXPECT_TRUE(succeeded(problem.verify()));

  ScopedDiagnosticHandler handler(&ir.context, [](Diagnostic &) {});
  auto truncated = *parallelPoint;
  truncated.schedule.operations.pop_back();
  EXPECT_TRUE(failed(applyResourceParetoPoint(problem, last, truncated)));

  auto wrongLatency = *parallelPoint;
  ++wrongLatency.latency;
  EXPECT_TRUE(failed(applyResourceParetoPoint(problem, last, wrongLatency)));
}

TEST(ResourceScheduleCertificateTest, ReappliesRotatingModuloParetoPoint) {
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
  NodeRLSchedulerOptions options;
  options.episodes = 8;
  options.seed = 7;
  auto frontier = exploreNodeRLPareto(problem, last, allocations, options);
  ASSERT_TRUE(succeeded(frontier));
  ASSERT_EQ(frontier->size(), 2u);
  EXPECT_EQ(*problem.getLimit(resource), 1u);
  EXPECT_EQ(*problem.getInitiationInterval(), 3u);
  for (const auto &point : *frontier) {
    ASSERT_TRUE(succeeded(applyResourceParetoPoint(problem, last, point)));
    EXPECT_EQ(*problem.getStartTime(last), point.latency);
    EXPECT_EQ(*problem.getInitiationInterval(), point.initiationInterval);
  }

  auto rotatingPoint = llvm::find_if(*frontier, [](const auto &point) {
    return point.allocation.limits.front().limit == 2;
  });
  ASSERT_NE(rotatingPoint, frontier->end());
  EXPECT_EQ(rotatingPoint->initiationInterval, 2u);
  ASSERT_EQ(rotatingPoint->schedule.operations.size(), 2u);
  const auto &operation = rotatingPoint->schedule.operations.front();
  EXPECT_FALSE(operation.resourceBindings);
  ASSERT_EQ(operation.rotatingReservations.size(), 1u);
  const auto &reservation = operation.rotatingReservations.front();
  EXPECT_TRUE(reservation.resource == resource);
  EXPECT_EQ(reservation.phase, operation.startTime % 2);
  EXPECT_EQ(reservation.period, 2u);
  EXPECT_EQ(reservation.hold, 3u);
  EXPECT_EQ(reservation.instances, 2u);

  ASSERT_TRUE(
      succeeded(applyResourceParetoPoint(problem, last, *rotatingPoint)));
  EXPECT_EQ(*problem.getLimit(resource), 2u);
  EXPECT_EQ(*problem.getInitiationInterval(), 2u);
  EXPECT_FALSE(problem.getResourceBindings(recurrenceOp));
  EXPECT_TRUE(succeeded(problem.verify()));

  ScopedDiagnosticHandler handler(&ir.context, [](Diagnostic &) {});
  auto wrongReservation = *rotatingPoint;
  ++wrongReservation.schedule.operations.front()
        .rotatingReservations.front()
        .hold;
  EXPECT_TRUE(
      failed(applyResourceParetoPoint(problem, last, wrongReservation)));
}

} // namespace
