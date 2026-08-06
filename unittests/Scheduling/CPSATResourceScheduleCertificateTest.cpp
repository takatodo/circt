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

#include <array>

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

static ::testing::AssertionResult
auditStaticFrontier(ModuloProblem &problem, Operation *last,
                    ArrayRef<ResourceParetoPoint> frontier,
                    unsigned expectedPoints, unsigned expectedOperations,
                    unsigned expectedResourceUses) {
  if (frontier.size() != expectedPoints)
    return ::testing::AssertionFailure()
           << "expected " << expectedPoints << " frontier points, got "
           << frontier.size();
  for (const auto &point : frontier) {
    if (point.schedule.operations.size() != expectedOperations)
      return ::testing::AssertionFailure()
             << "certificate operation count differs at cost "
             << point.resourceCost;
    if (failed(applyResourceParetoPoint(problem, last, point)))
      return ::testing::AssertionFailure()
             << "could not apply frontier point at cost " << point.resourceCost;
    if (failed(problem.verify()))
      return ::testing::AssertionFailure()
             << "frontier point failed verification at cost "
             << point.resourceCost;
    if (*problem.getInitiationInterval() != point.initiationInterval ||
        *problem.getStartTime(last) != point.latency)
      return ::testing::AssertionFailure()
             << "replay changed metrics at cost " << point.resourceCost;

    unsigned staticBindings = 0;
    unsigned rotatingReservations = 0;
    for (const auto &operation : point.schedule.operations) {
      if (operation.resourceBindings)
        staticBindings += operation.resourceBindings->size();
      rotatingReservations += operation.rotatingReservations.size();
    }
    if (staticBindings != expectedResourceUses || rotatingReservations != 0)
      return ::testing::AssertionFailure()
             << "expected a complete static coloring at cost "
             << point.resourceCost << ", got " << staticBindings
             << " bindings and " << rotatingReservations
             << " rotating reservations";
  }
  return ::testing::AssertionSuccess();
}

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

TEST(CPSATResourceScheduleCertificateTest,
     AuditsComplexNodeRLFrontierAndCPSATPoint) {
  TestIR ir;
  ModuloProblem problem(*ir.module);
  auto multiply = problem.getOrInsertOperatorType("multiply");
  auto add = problem.getOrInsertOperatorType("add");
  auto sink = problem.getOrInsertOperatorType("sink");
  problem.setLatency(multiply, 2);
  problem.setLatency(add, 1);
  problem.setLatency(sink, 1);
  auto multiplier = problem.getOrInsertResourceType("multiplier");
  auto divider = problem.getOrInsertResourceType("divider");
  problem.setResourceInitiationInterval(multiplier, 3);
  problem.setResourceInitiationInterval(divider, 2);
  problem.setResourceCost(multiplier, 3);
  problem.setResourceCost(divider, 5);

  struct Tile {
    std::array<Operation *, 2> products;
    std::array<Operation *, 2> temporaries;
    std::array<Operation *, 2> projected;
    std::array<Operation *, 2> outputs;
  };
  std::array<Tile, 4> tiles;
  auto configure = [&](Operation *op, Problem::OperatorType operatorType,
                       Problem::ResourceType resource) {
    problem.insertOperation(op);
    problem.setLinkedOperatorType(op, operatorType);
    problem.setLinkedResourceTypes(op, {resource});
  };
  for (auto &tile : tiles) {
    for (unsigned index = 0; index != 2; ++index) {
      tile.products[index] = ir.createOperation();
      configure(tile.products[index], multiply, multiplier);
      tile.temporaries[index] = ir.createOperation();
      configure(tile.temporaries[index], add, divider);
      tile.projected[index] = ir.createOperation();
      configure(tile.projected[index], multiply, multiplier);
      tile.outputs[index] = ir.createOperation();
      configure(tile.outputs[index], add, divider);
    }
  }
  auto *last = ir.createOperation();
  problem.insertOperation(last);
  problem.setLinkedOperatorType(last, sink);

  auto addDependence = [&](Operation *source, Operation *destination,
                           unsigned distance = 0) {
    Problem::Dependence dependence(source, destination);
    if (failed(problem.insertDependence(dependence)))
      return failure();
    if (distance != 0)
      problem.setDistance(dependence, distance);
    return success();
  };
  for (const auto &tile : tiles) {
    ASSERT_TRUE(
        succeeded(addDependence(tile.products[0], tile.temporaries[0])));
    ASSERT_TRUE(
        succeeded(addDependence(tile.temporaries[1], tile.temporaries[0], 1)));
    ASSERT_TRUE(
        succeeded(addDependence(tile.temporaries[0], tile.projected[0])));
    ASSERT_TRUE(succeeded(addDependence(tile.projected[0], tile.outputs[0])));
    ASSERT_TRUE(succeeded(addDependence(tile.outputs[1], tile.outputs[0], 1)));

    ASSERT_TRUE(
        succeeded(addDependence(tile.products[1], tile.temporaries[1])));
    ASSERT_TRUE(
        succeeded(addDependence(tile.temporaries[0], tile.temporaries[1])));
    ASSERT_TRUE(
        succeeded(addDependence(tile.temporaries[1], tile.projected[1])));
    ASSERT_TRUE(succeeded(addDependence(tile.projected[1], tile.outputs[1])));
    ASSERT_TRUE(succeeded(addDependence(tile.outputs[0], tile.outputs[1])));
    ASSERT_TRUE(succeeded(addDependence(tile.outputs[1], last)));
  }
  ASSERT_TRUE(succeeded(problem.check()));

  SmallVector<ResourceAllocation> allocations;
  for (unsigned multiplierLimit : {1u, 2u, 4u})
    for (unsigned dividerLimit : {1u, 2u})
      allocations.push_back(
          {{{multiplier, multiplierLimit}, {divider, dividerLimit}}});

  NodeRLSchedulerOptions nodeRLOptions;
  nodeRLOptions.episodes = 32;
  nodeRLOptions.seed = 0;
  auto nodeRLFrontier =
      exploreNodeRLPareto(problem, last, allocations, nodeRLOptions);
  ASSERT_TRUE(succeeded(nodeRLFrontier));
  EXPECT_TRUE(auditStaticFrontier(problem, last, *nodeRLFrontier, 5, 33, 32));

  CPSATSchedulerOptions cpsatOptions;
  cpsatOptions.timeLimitSeconds = 10.0;
  cpsatOptions.numWorkers = 4;
  ArrayRef<ResourceAllocation> exactAllocation(allocations.back());
  auto cpsatFrontier =
      exploreCPSATPareto(problem, last, exactAllocation, cpsatOptions);
  ASSERT_TRUE(succeeded(cpsatFrontier));
  EXPECT_TRUE(auditStaticFrontier(problem, last, *cpsatFrontier, 1, 33, 32));

  SmallVector<std::tuple<uint64_t, unsigned, unsigned>> exactMetrics;
  for (const auto &point : *cpsatFrontier)
    exactMetrics.emplace_back(point.resourceCost, point.initiationInterval,
                              point.latency);
  llvm::sort(exactMetrics);
  SmallVector<std::tuple<uint64_t, unsigned, unsigned>> expectedMetrics = {
      {22, 16, 17}};
  EXPECT_EQ(exactMetrics, expectedMetrics);
}

} // namespace
