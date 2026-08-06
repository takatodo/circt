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
  ResourceParetoPoint parallelPoint;
  parallelPoint.allocation = allocations[1];
  parallelPoint.latency = 1;
  parallelPoint.resourceCost = 2;
  parallelPoint.schedule.operations.resize(3);
  parallelPoint.schedule.operations[0].startTime = 0;
  parallelPoint.schedule.operations[0].resourceBindings =
      SmallVector<unsigned, 2>{0};
  parallelPoint.schedule.operations[1].startTime = 0;
  parallelPoint.schedule.operations[1].resourceBindings =
      SmallVector<unsigned, 2>{1};
  parallelPoint.schedule.operations[2].startTime = 1;

  ASSERT_TRUE(succeeded(applyResourceParetoPoint(problem, last, parallelPoint)));
  EXPECT_EQ(*problem.getLimit(resource), 2u);
  EXPECT_EQ(*problem.getStartTime(last), parallelPoint.latency);
  EXPECT_TRUE(succeeded(problem.verify()));

  ScopedDiagnosticHandler handler(&ir.context, [](Diagnostic &) {});
  auto truncated = parallelPoint;
  truncated.schedule.operations.pop_back();
  EXPECT_TRUE(failed(applyResourceParetoPoint(problem, last, truncated)));

  auto wrongLatency = parallelPoint;
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
  ResourceParetoPoint rotatingPoint;
  rotatingPoint.allocation = allocations[1];
  rotatingPoint.latency = 1;
  rotatingPoint.resourceCost = 2;
  rotatingPoint.initiationInterval = 2;
  rotatingPoint.schedule.operations.resize(2);
  rotatingPoint.schedule.operations[0].startTime = 0;
  rotatingPoint.schedule.operations[0].rotatingReservations.push_back(
      {resource, 0, 2, 3, 2});
  rotatingPoint.schedule.operations[1].startTime = 1;

  const auto &operation = rotatingPoint.schedule.operations.front();
  EXPECT_FALSE(operation.resourceBindings);
  ASSERT_EQ(operation.rotatingReservations.size(), 1u);
  const auto &reservation = operation.rotatingReservations.front();
  EXPECT_TRUE(reservation.resource == resource);
  EXPECT_EQ(reservation.phase, operation.startTime % 2);
  EXPECT_EQ(reservation.period, 2u);
  EXPECT_EQ(reservation.hold, 3u);
  EXPECT_EQ(reservation.instances, 2u);

  ASSERT_TRUE(succeeded(applyResourceParetoPoint(problem, last, rotatingPoint)));
  EXPECT_EQ(*problem.getLimit(resource), 2u);
  EXPECT_EQ(*problem.getInitiationInterval(), 2u);
  EXPECT_FALSE(problem.getResourceBindings(recurrenceOp));
  EXPECT_TRUE(succeeded(problem.verify()));

  ScopedDiagnosticHandler handler(&ir.context, [](Diagnostic &) {});
  auto wrongReservation = rotatingPoint;
  ++wrongReservation.schedule.operations.front()
        .rotatingReservations.front()
        .hold;
  EXPECT_TRUE(
      failed(applyResourceParetoPoint(problem, last, wrongReservation)));
}

} // namespace
