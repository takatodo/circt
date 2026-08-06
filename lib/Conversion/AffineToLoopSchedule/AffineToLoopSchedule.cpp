//===- AffineToLoopSchedule.cpp--------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/AffineToLoopSchedule.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/LoopSchedule/LoopScheduleOps.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cmath>
#include <limits>

#define DEBUG_TYPE "affine-to-loopschedule"

namespace circt {
#define GEN_PASS_DEF_AFFINETOLOOPSCHEDULE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace mlir::func;
using namespace mlir::affine;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::loopschedule;

namespace {

static constexpr StringLiteral resourceBindingsAttrName =
    "circt.resource_bindings";
static constexpr StringLiteral rotatingReservationsAttrName =
    "circt.rotating_resource_reservations";

#ifdef SCHEDULING_OR_TOOLS
static StringRef getCPSATStatusName(CPSATSolveStatus status) {
  switch (status) {
  case CPSATSolveStatus::optimal:
    return "optimal";
  case CPSATSolveStatus::feasible:
    return "feasible";
  case CPSATSolveStatus::infeasible:
    return "infeasible";
  case CPSATSolveStatus::unknown:
    return "unknown";
  }
  llvm_unreachable("unknown CP-SAT solve status");
}

static StringRef
getCPSATResourceModelName(CPSATModuloResourceModel resourceModel) {
  switch (resourceModel) {
  case CPSATModuloResourceModel::autoSelect:
    return "auto";
  case CPSATModuloResourceModel::oneHot:
    return "onehot";
  case CPSATModuloResourceModel::cumulative:
    return "cumulative";
  }
  llvm_unreachable("unknown CP-SAT resource model");
}
#endif

/// Build a periodic instance-selection table for one unbound resource. The
/// table is represented by one color per operation and iteration class. A
/// selector period of one is a static allocation; longer periods rotate the
/// selected instance between loop iterations. Mixing this scheme with
/// pre-existing static bindings needs a combined allocator.
struct RotatingSelectorAllocation {
  DenseMap<Operation *, SmallVector<unsigned>> selectors;
  unsigned instances = 0;
};

static std::optional<RotatingSelectorAllocation>
buildRotatingSelectorTable(ModuloProblem &problem,
                           Problem::ResourceType resource) {
  unsigned instanceLimit = problem.getLimit(resource).value_or(0);
  unsigned period = *problem.getInitiationInterval();
  unsigned hold = problem.getResourceInitiationInterval(resource).value_or(1);
  if (instanceLimit == 0)
    return std::nullopt;

  SmallVector<Operation *> users;
  for (auto *op : problem.getOperations()) {
    auto resources = problem.getLinkedResourceTypes(op);
    if (resources && llvm::is_contained(*resources, resource)) {
      if (problem.getResourceBindings(op))
        return std::nullopt;
      users.push_back(op);
    }
  }
  if (users.empty())
    return std::nullopt;

  llvm::sort(users, [&](Operation *lhs, Operation *rhs) {
    return std::make_tuple(*problem.getStartTime(lhs) % period, lhs) <
           std::make_tuple(*problem.getStartTime(rhs) % period, rhs);
  });

  // Find the shortest repeating selector table. The number of available
  // instances is an upper bound, not a reason to lengthen the selector or
  // materialize unused cells. Trying all periods through the instance limit
  // includes the previous fixed-period construction while preferring smaller
  // hardware/control realizations.
  for (unsigned selectorPeriod = 1; selectorPeriod <= instanceLimit;
       ++selectorPeriod) {
    uint64_t horizon64 = uint64_t(period) * selectorPeriod;
    if (horizon64 == 0 || horizon64 > 4096)
      break;
    // A request would overlap the next repetition of itself.
    if (hold > horizon64)
      continue;
    unsigned horizon = horizon64;
    SmallVector<SmallVector<bool>> occupied(instanceLimit,
                                            SmallVector<bool>(horizon, false));
    DenseMap<Operation *, SmallVector<unsigned>> table;
    unsigned usedInstances = 0;
    bool failed = false;
    for (auto *op : users) {
      auto &selectors = table[op];
      for (unsigned iterationClass = 0; iterationClass != selectorPeriod;
           ++iterationClass) {
        unsigned start =
            (*problem.getStartTime(op) % period) + iterationClass * period;
        unsigned instance = 0;
        for (; instance != instanceLimit; ++instance) {
          bool available = true;
          for (unsigned offset = 0; offset != hold; ++offset)
            if (occupied[instance][(start + offset) % horizon]) {
              available = false;
              break;
            }
          if (available)
            break;
        }
        if (instance == instanceLimit) {
          failed = true;
          break;
        }
        for (unsigned offset = 0; offset != hold; ++offset)
          occupied[instance][(start + offset) % horizon] = true;
        selectors.push_back(instance);
        usedInstances = std::max(usedInstances, instance + 1);
      }
      if (failed)
        break;
    }
    if (!failed)
      return RotatingSelectorAllocation{std::move(table), usedInstances};
  }
  return std::nullopt;
}

struct AffineToLoopSchedule
    : public circt::impl::AffineToLoopScheduleBase<AffineToLoopSchedule> {
  void runOnOperation() override;

private:
  ModuloProblem getModuloProblem(CyclicProblem &prob);
  LogicalResult
  lowerAffineStructures(MemoryDependenceAnalysis &dependenceAnalysis);
  LogicalResult populateOperatorTypes(SmallVectorImpl<AffineForOp> &loopNest,
                                      ModuloProblem &problem);
  LogicalResult solveSchedulingProblem(SmallVectorImpl<AffineForOp> &loopNest,
                                       ModuloProblem &problem);
  LogicalResult
  createLoopSchedulePipeline(SmallVectorImpl<AffineForOp> &loopNest,
                             ModuloProblem &problem);

  CyclicSchedulingAnalysis *schedulingAnalysis;
};

} // namespace

ModuloProblem AffineToLoopSchedule::getModuloProblem(CyclicProblem &prob) {
  ModuloProblem modProb(prob.getContainingOp());
  for (auto *op : prob.getOperations()) {
    auto opr = prob.getLinkedOperatorType(op);
    if (opr.has_value()) {
      modProb.setLinkedOperatorType(op, opr.value());
      auto latency = prob.getLatency(opr.value());
      if (latency.has_value())
        modProb.setLatency(opr.value(), latency.value());
    }
    auto rsrc = prob.getLinkedResourceTypes(op);
    if (rsrc.has_value())
      modProb.setLinkedResourceTypes(op, rsrc.value());
    modProb.insertOperation(op);
  }

  for (auto *op : prob.getOperations()) {
    for (auto dep : prob.getDependences(op)) {
      if (dep.isAuxiliary()) {
        auto depInserted = modProb.insertDependence(dep);
        assert(succeeded(depInserted));
        (void)depInserted;
      }
      auto distance = prob.getDistance(dep);
      if (distance.has_value())
        modProb.setDistance(dep, distance.value());
    }
  }

  return modProb;
}

void AffineToLoopSchedule::runOnOperation() {
  // Get dependence analysis for the whole function.
  auto &dependenceAnalysis = getAnalysis<MemoryDependenceAnalysis>();

  // After dependence analysis, materialize affine structures.
  if (failed(lowerAffineStructures(dependenceAnalysis)))
    return signalPassFailure();

  // Get scheduling analysis for the whole function.
  schedulingAnalysis = &getAnalysis<CyclicSchedulingAnalysis>();

  // Collect perfectly nested loops and work on them.
  auto outerLoops = getOperation().getOps<AffineForOp>();
  for (auto root : llvm::make_early_inc_range(outerLoops)) {
    SmallVector<AffineForOp> nestedLoops;
    getPerfectlyNestedLoops(nestedLoops, root);

    // Restrict to single loops to simplify things for now.
    if (nestedLoops.size() != 1)
      continue;

    ModuloProblem moduloProblem =
        getModuloProblem(schedulingAnalysis->getProblem(nestedLoops.back()));

    // Populate the target operator types.
    if (failed(populateOperatorTypes(nestedLoops, moduloProblem)))
      return signalPassFailure();

    // Solve the scheduling problem computed by the analysis.
    if (failed(solveSchedulingProblem(nestedLoops, moduloProblem)))
      return signalPassFailure();

    // Convert the IR.
    if (failed(createLoopSchedulePipeline(nestedLoops, moduloProblem)))
      return signalPassFailure();
  }
}

/// Apply the affine map from an 'affine.load' operation to its operands, and
/// feed the results to a newly created 'memref.load' operation (which replaces
/// the original 'affine.load').
/// Also replaces the affine load with the memref load in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineLoadLowering : public OpConversionPattern<AffineLoadOp> {
public:
  AffineLoadLowering(MLIRContext *context,
                     MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineLoadOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto resultOperands =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!resultOperands)
      return failure();

    // Build memref.load memref[expandedMap.results].
    auto memrefLoad = rewriter.replaceOpWithNewOp<memref::LoadOp>(
        op, op.getMemRef(), *resultOperands);

    dependenceAnalysis.replaceOp(op, memrefLoad);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Apply the affine map from an 'affine.store' operation to its operands, and
/// feed the results to a newly created 'memref.store' operation (which replaces
/// the original 'affine.store').
/// Also replaces the affine store with the memref store in dependenceAnalysis.
/// TODO(mikeurbach): this is copied from AffineToStandard, see if we can reuse.
class AffineStoreLowering : public OpConversionPattern<AffineStoreOp> {
public:
  AffineStoreLowering(MLIRContext *context,
                      MemoryDependenceAnalysis &dependenceAnalysis)
      : OpConversionPattern(context), dependenceAnalysis(dependenceAnalysis) {}

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Expand affine map from 'affineStoreOp'.
    SmallVector<Value, 8> indices(op.getMapOperands());
    auto maybeExpandedMap =
        expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
    if (!maybeExpandedMap)
      return failure();

    // Build memref.store valueToStore, memref[expandedMap.results].
    auto memrefStore = rewriter.replaceOpWithNewOp<memref::StoreOp>(
        op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);

    dependenceAnalysis.replaceOp(op, memrefStore);

    return success();
  }

private:
  MemoryDependenceAnalysis &dependenceAnalysis;
};

/// Helper to hoist computation out of scf::IfOp branches, turning it into a
/// mux-like operation, and exposing potentially concurrent execution of its
/// branches.
struct IfOpHoisting : OpConversionPattern<IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&]() {
      if (!op.thenBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.thenBlock(), --op.thenBlock()->end());
        rewriter.inlineBlockBefore(&op.getThenRegion().front(), op);
      }
      if (op.elseBlock() && !op.elseBlock()->without_terminator().empty()) {
        rewriter.splitBlock(op.elseBlock(), --op.elseBlock()->end());
        rewriter.inlineBlockBefore(&op.getElseRegion().front(), op);
      }
    });

    return success();
  }
};

/// Helper to determine if an scf::IfOp is in mux-like form.
static bool ifOpLegalityCallback(IfOp op) {
  return op.thenBlock()->without_terminator().empty() &&
         (!op.elseBlock() || op.elseBlock()->without_terminator().empty());
}

/// Helper to mark AffineYieldOp legal, unless it is inside a partially
/// converted scf::IfOp.
static bool yieldOpLegalityCallback(AffineYieldOp op) {
  return !op->getParentOfType<IfOp>();
}

/// After analyzing memory dependences, and before creating the schedule, we
/// want to materialize affine operations with arithmetic, scf, and memref
/// operations, which make the condition computation of addresses, etc.
/// explicit. This is important so the schedule can consider potentially complex
/// computations in the condition of ifs, or the addresses of loads and stores.
/// The dependence analysis will be updated so the dependences from the affine
/// loads and stores are now on the memref loads and stores.
LogicalResult AffineToLoopSchedule::lowerAffineStructures(
    MemoryDependenceAnalysis &dependenceAnalysis) {
  auto *context = &getContext();
  auto op = getOperation();

  ConversionTarget target(*context);
  target.addLegalDialect<AffineDialect, ArithDialect, MemRefDialect,
                         SCFDialect>();
  target.addIllegalOp<AffineIfOp, AffineLoadOp, AffineStoreOp>();
  target.addDynamicallyLegalOp<IfOp>(ifOpLegalityCallback);
  target.addDynamicallyLegalOp<AffineYieldOp>(yieldOpLegalityCallback);

  RewritePatternSet patterns(context);
  populateAffineToStdConversionPatterns(patterns);
  patterns.add<AffineLoadLowering>(context, dependenceAnalysis);
  patterns.add<AffineStoreLowering>(context, dependenceAnalysis);
  patterns.add<IfOpHoisting>(context);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    return failure();

  return success();
}

/// Populate the schedling problem operator types for the dialect we are
/// targetting. Right now, we assume Calyx, which has a standard library with
/// well-defined operator latencies. Ultimately, we should move this to a
/// dialect interface in the Scheduling dialect.
LogicalResult AffineToLoopSchedule::populateOperatorTypes(
    SmallVectorImpl<AffineForOp> &loopNest, ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  Problem::OperatorType combOpr = problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);
  Problem::OperatorType seqOpr = problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);
  Problem::OperatorType mcOpr = problem.getOrInsertOperatorType("multicycle");
  problem.setLatency(mcOpr, 3);

  Operation *unsupported;
  WalkResult result = forOp.getBody()->walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AddIOp, IfOp, AffineYieldOp, arith::ConstantOp, CmpIOp,
              IndexCastOp, memref::AllocaOp, YieldOp>([&](Operation *combOp) {
          // Some known combinational ops.
          problem.setLinkedOperatorType(combOp, combOpr);
          return WalkResult::advance();
        })
        .Case<AddIOp, CmpIOp>([&](Operation *seqOp) {
          // These ops need to be sequential for now because we do not
          // have enough information to chain them together yet.
          problem.setLinkedOperatorType(seqOp, seqOpr);
          return WalkResult::advance();
        })
        .Case<AffineStoreOp, memref::StoreOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineStoreOp>(*memOp)
                             ? cast<AffineStoreOp>(*memOp).getMemRef()
                             : cast<memref::StoreOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);

          auto memRsrc = problem.getOrInsertResourceType(
              "mem_" + std::to_string(hash_value(memRef)) + "_rsrc");
          problem.setLimit(memRsrc, 1);
          problem.setLinkedResourceTypes(
              memOp, SmallVector<Problem::ResourceType>{memRsrc});

          return WalkResult::advance();
        })
        .Case<AffineLoadOp, memref::LoadOp>([&](Operation *memOp) {
          // Some known sequential ops. In certain cases, reads may be
          // combinational in Calyx, but taking advantage of that is left as
          // a future enhancement.
          Value memRef = isa<AffineLoadOp>(*memOp)
                             ? cast<AffineLoadOp>(*memOp).getMemRef()
                             : cast<memref::LoadOp>(*memOp).getMemRef();
          Problem::OperatorType memOpr = problem.getOrInsertOperatorType(
              "mem_" + std::to_string(hash_value(memRef)));
          problem.setLatency(memOpr, 1);
          problem.setLinkedOperatorType(memOp, memOpr);

          auto memRsrc = problem.getOrInsertResourceType(
              "mem_" + std::to_string(hash_value(memRef)) + "_rsrc");
          problem.setLimit(memRsrc, 1);
          problem.setLinkedResourceTypes(
              memOp, SmallVector<Problem::ResourceType>{memRsrc});

          return WalkResult::advance();
        })
        .Case<MulIOp>([&](Operation *mcOp) {
          // Some known multi-cycle ops.
          problem.setLinkedOperatorType(mcOp, mcOpr);
          if (multiplierLimit != 0) {
            auto mulRsrc = problem.getOrInsertResourceType("multiplier");
            problem.setLimit(mulRsrc, multiplierLimit);
            problem.setResourceInitiationInterval(mulRsrc, multiplierII);
            problem.setLinkedResourceTypes(
                mcOp, SmallVector<Problem::ResourceType>{mulRsrc});
          }
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return forOp.emitError("unsupported operation ") << *unsupported;

  return success();
}

/// Solve the pre-computed scheduling problem.
LogicalResult AffineToLoopSchedule::solveSchedulingProblem(
    SmallVectorImpl<AffineForOp> &loopNest, ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  // Optionally debug problem inputs.
  LLVM_DEBUG(forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
    llvm::dbgs() << "Scheduling inputs for " << *op;
    auto opr = problem.getLinkedOperatorType(op);
    llvm::dbgs() << "\n  opr = " << opr->getAttr();
    llvm::dbgs() << "\n  latency = " << problem.getLatency(*opr);
    for (auto dep : problem.getDependences(op))
      if (dep.isAuxiliary())
        llvm::dbgs() << "\n  dep = { distance = " << problem.getDistance(dep)
                     << ", source = " << *dep.getSource() << " }";
    llvm::dbgs() << "\n\n";
  }));

  // Verify and solve the problem.
  if (failed(problem.check()))
    return failure();

  auto *anchor = forOp.getBody()->getTerminator();
  if (scheduler == "simplex") {
    if (failed(scheduleSimplex(problem, anchor)))
      return failure();
#ifdef SCHEDULING_OR_TOOLS
  } else if (scheduler == "cpsat") {
    double timeLimit = cpsatTimeLimit;
    double balancedProbeTimeLimit = cpsatBalancedProbeTimeLimit;
    if (!std::isfinite(timeLimit) || timeLimit < 0.0)
      return forOp.emitError("invalid cpsat-time-limit; expected a finite "
                             "non-negative number");
    if (!std::isfinite(balancedProbeTimeLimit) || balancedProbeTimeLimit <= 0.0)
      return forOp.emitError("invalid cpsat-balanced-probe-time-limit; "
                             "expected a finite positive number");

    CPSATModuloResourceModel resourceModel;
    if (cpsatResourceModel == "auto")
      resourceModel = CPSATModuloResourceModel::autoSelect;
    else if (cpsatResourceModel == "onehot")
      resourceModel = CPSATModuloResourceModel::oneHot;
    else if (cpsatResourceModel == "cumulative")
      resourceModel = CPSATModuloResourceModel::cumulative;
    else
      return forOp.emitError("invalid cpsat-resource-model; expected auto, "
                             "onehot, or cumulative");

    CPSATSchedulerOptions options;
    options.timeLimitSeconds = timeLimit;
    options.numWorkers = cpsatWorkers;
    options.minimizeLatency = cpsatMinimizeLatency;
    options.enableBalancedProbe = cpsatBalancedProbe;
    options.balancedProbeTimeLimitSeconds = balancedProbeTimeLimit;
    options.moduloResourceModel = resourceModel;
    CPSATSchedulerResult result;
    if (failed(scheduleCPSAT(problem, anchor, options, &result)))
      return failure();
    if (cpsatReportStatistics)
      llvm::errs() << "affine-cpsat: status="
                   << getCPSATStatusName(result.status)
                   << ", lower-bound=" << result.lowerBound
                   << ", II=" << result.initiationInterval
                   << ", resource-model="
                   << getCPSATResourceModelName(result.moduloResourceModel)
                   << ", phase-indicators=" << result.phaseIndicatorCount
                   << ", balanced-probe="
                   << (!options.enableBalancedProbe    ? "off"
                       : result.balancedProbeSucceeded ? "hit"
                       : result.balancedProbeAttempted ? "miss"
                                                       : "skipped")
                   << "\n";
#endif
  } else {
    return forOp.emitError() << "unsupported modulo scheduler '" << scheduler
                             << "'; expected 'simplex'"
#ifdef SCHEDULING_OR_TOOLS
                             << " or 'cpsat'";
#else
                             << " (or 'cpsat' in an OR-Tools build)";
#endif
  }

  // Verify the solution.
  if (failed(problem.verify()))
    return failure();

  // Optionally debug problem outputs.
  LLVM_DEBUG({
    llvm::dbgs() << "Scheduled initiation interval = "
                 << problem.getInitiationInterval() << "\n\n";
    forOp.getBody()->walk<WalkOrder::PreOrder>([&](Operation *op) {
      llvm::dbgs() << "Scheduling outputs for " << *op;
      llvm::dbgs() << "\n  start = " << problem.getStartTime(op);
      llvm::dbgs() << "\n\n";
    });
  });

  return success();
}

/// Create the loopschedule pipeline op for a loop nest.
LogicalResult AffineToLoopSchedule::createLoopSchedulePipeline(
    SmallVectorImpl<AffineForOp> &loopNest, ModuloProblem &problem) {
  // Scheduling analyis only considers the innermost loop nest for now.
  auto forOp = loopNest.back();

  auto outerLoop = loopNest.front();
  auto innerLoop = loopNest.back();
  ImplicitLocOpBuilder builder(outerLoop.getLoc(), outerLoop);

  // Create Values for the loop's lower and upper bounds.
  Value lowerBound = lowerAffineLowerBound(innerLoop, builder);
  Value upperBound = lowerAffineUpperBound(innerLoop, builder);
  int64_t stepValue = innerLoop.getStep().getSExtValue();
  auto step = arith::ConstantOp::create(
      builder, IntegerAttr::get(builder.getIndexType(), stepValue));

  // Create the pipeline op, with the same result types as the inner loop. An
  // iter arg is created for the induction variable.
  TypeRange resultTypes = innerLoop.getResultTypes();

  auto ii = builder.getI64IntegerAttr(problem.getInitiationInterval().value());

  SmallVector<Value> iterArgs;
  iterArgs.push_back(lowerBound);
  iterArgs.append(innerLoop.getInits().begin(), innerLoop.getInits().end());

  // If possible, attach a constant trip count attribute. This could be
  // generalized to support non-constant trip counts by supporting an AffineMap.
  std::optional<IntegerAttr> tripCountAttr;
  if (auto tripCount = forOp.getStaticTripCount())
    tripCountAttr = builder.getI64IntegerAttr(tripCount->getZExtValue());

  auto pipeline = LoopSchedulePipelineOp::create(builder, resultTypes, ii,
                                                 tripCountAttr, iterArgs);

  // Create the condition, which currently just compares the induction variable
  // to the upper bound.
  Block &condBlock = pipeline.getCondBlock();
  builder.setInsertionPointToStart(&condBlock);
  auto cmpResult = arith::CmpIOp::create(builder, builder.getI1Type(),
                                         arith::CmpIPredicate::ult,
                                         condBlock.getArgument(0), upperBound);
  condBlock.getTerminator()->insertOperands(0, {cmpResult});

  // Add the non-yield operations to their start time groups.
  DenseMap<unsigned, SmallVector<Operation *>> startGroups;
  for (auto *op : problem.getOperations()) {
    if (isa<AffineYieldOp, YieldOp>(op))
      continue;
    auto startTime = problem.getStartTime(op);
    startGroups[*startTime].push_back(op);
  }

  // Compute each cyclic selector table once. Besides avoiding repeated graph
  // coloring for every operation, this materializes an implementable instance
  // selection for schedulers such as CP-SAT that only return start times.
  DenseMap<Problem::ResourceType, RotatingSelectorAllocation>
      rotatingSelectorTables;
  DenseSet<Problem::ResourceType> visitedResources;
  for (auto *op : problem.getOperations()) {
    auto resources = problem.getLinkedResourceTypes(op);
    if (!resources)
      continue;
    for (auto resource : *resources) {
      if (!visitedResources.insert(resource).second)
        continue;
      if (auto table = buildRotatingSelectorTable(problem, resource))
        rotatingSelectorTables.try_emplace(resource, std::move(*table));
    }
  }

  // Maintain mappings of values in the loop body and results of stages,
  // initially populated with the iter args.
  IRMapping valueMap;
  // Nested loops are not supported yet.
  assert(iterArgs.size() == forOp.getBody()->getNumArguments());
  for (size_t i = 0; i < iterArgs.size(); ++i)
    valueMap.map(forOp.getBody()->getArgument(i),
                 pipeline.getStagesBlock().getArgument(i));

  // Create the stages.
  Block &stagesBlock = pipeline.getStagesBlock();
  builder.setInsertionPointToStart(&stagesBlock);

  // Start with the operation-bearing stages. Forwarding-only stages are added
  // below once the required register lifetimes are known.
  SmallVector<unsigned> operationStartTimes;
  for (const auto &group : startGroups)
    operationStartTimes.push_back(group.first);
  llvm::sort(operationStartTimes);

  DominanceInfo dom(getOperation());

  // Keys for translating values in each stage
  SmallVector<SmallVector<Value>> registerValues;
  SmallVector<SmallVector<Type>> registerTypes;

  // The maps that ensure a stage uses the correct version of a value
  SmallVector<IRMapping> stageValueMaps;

  // For storing the range of stages an operation's results need to be valid for
  DenseMap<Operation *, std::pair<unsigned, unsigned>> pipeTimes;

  for (auto startTime : operationStartTimes) {
    auto group = startGroups[startTime];

    // Collect the return types for this stage. Operations whose results are not
    // used within this stage are returned.
    auto isLoopTerminator = [forOp](Operation *op) {
      return isa<AffineYieldOp>(op) && op->getParentOp() == forOp;
    };

    // Initialize set of registers up until this point in time
    for (unsigned i = registerValues.size(); i <= startTime; ++i)
      registerValues.emplace_back(SmallVector<Value>());

    // Check each operation to see if its results need plumbing
    for (auto *op : group) {
      if (op->getUsers().empty())
        continue;

      unsigned pipeEndTime = 0;
      for (auto *user : op->getUsers()) {
        unsigned userStartTime = *problem.getStartTime(user);
        if (*problem.getStartTime(user) > startTime)
          pipeEndTime = std::max(pipeEndTime, userStartTime);
        else if (isLoopTerminator(user))
          // Manually forward the value into the terminator's valueMap
          pipeEndTime = std::max(pipeEndTime, userStartTime + 1);
      }

      // Insert the range of pipeline stages the value needs to be valid for
      pipeTimes[op] = std::pair(startTime, pipeEndTime);

      // Add register stages for each time slice we need to pipe to
      for (unsigned i = registerValues.size(); i <= pipeEndTime; ++i)
        registerValues.push_back(SmallVector<Value>());

      // Keep a collection of this stages results as keys to our valueMaps
      for (auto result : op->getResults())
        registerValues[startTime].push_back(result);

      // Other stages that use the value will need these values as keys too
      unsigned firstUse = std::max(
          startTime + 1,
          startTime + *problem.getLatency(*problem.getLinkedOperatorType(op)));
      for (unsigned i = firstUse; i < pipeEndTime; ++i) {
        for (auto result : op->getResults())
          registerValues[i].push_back(result);
      }
    }
  }

  // Now make register Types and stageValueMaps
  for (unsigned i = 0; i < registerValues.size(); ++i) {
    SmallVector<mlir::Type> types;
    for (auto val : registerValues[i])
      types.push_back(val.getType());

    registerTypes.push_back(types);
    stageValueMaps.push_back(valueMap);
  }

  // One more map is needed for the pipeline stages terminator
  stageValueMaps.push_back(valueMap);

  // A value with a multi-cycle lifetime must pass through every physical
  // pipeline stage, including stages that contain no scheduled operation. If
  // these forwarding stages are omitted, the value map for the later consumer
  // remains empty (for example, the intermediate product in a 2MM kernel).
  SmallVector<unsigned> startTimes = operationStartTimes;
  for (auto [time, values] : llvm::enumerate(registerValues))
    if (!values.empty() && !llvm::is_contained(startTimes, time))
      startTimes.push_back(time);
  llvm::sort(startTimes);

  // Create stages along with maps
  for (auto startTime : startTimes) {
    auto group = startGroups[startTime];
    llvm::sort(group, [&](Operation *a, Operation *b) {
      if (a == b)
        return false;
      // Values defined and used in the same stage must be cloned in def-use
      // order. Dominance alone is not a strict ordering for sibling
      // operations, which made an affine 2MM inner loop occasionally clone a
      // consumer before its producer and leave a null mapped operand.
      if (llvm::is_contained(a->getUsers(), b))
        return true;
      if (llvm::is_contained(b->getUsers(), a))
        return false;
      if (dom.properlyDominates(a, b))
        return true;
      if (dom.properlyDominates(b, a))
        return false;
      if (a->getBlock() == b->getBlock())
        return a->isBeforeInBlock(b);
      return a < b;
    });
    auto stageTypes = registerTypes[startTime];
    // Add the induction variable increment in the first stage.
    if (startTime == 0)
      stageTypes.push_back(lowerBound.getType());

    // Create the stage itself.
    builder.setInsertionPoint(stagesBlock.getTerminator());
    auto startTimeAttr = builder.getIntegerAttr(
        builder.getIntegerType(64, /*isSigned=*/true), startTime);
    auto stage =
        LoopSchedulePipelineStageOp::create(builder, stageTypes, startTimeAttr);
    auto &stageBlock = stage.getBodyBlock();
    auto *stageTerminator = stageBlock.getTerminator();
    builder.setInsertionPointToStart(&stageBlock);

    for (auto *op : group) {
      auto *newOp = builder.clone(*op, stageValueMaps[startTime]);

      // Preserve the resource-to-instance assignment as target-neutral
      // metadata. A downstream implementation lowering can use it to share a
      // physical operator cell across mutually exclusive operations.
      if (auto bindings = problem.getResourceBindings(op)) {
        SmallVector<Attribute> bindingAttrs;
        SmallVector<Problem::ResourceType> limitedResources;
        if (auto resources = problem.getLinkedResourceTypes(op))
          for (auto resource : *resources)
            if (problem.getLimit(resource).value_or(0) > 0 &&
                !llvm::is_contained(limitedResources, resource))
              limitedResources.push_back(resource);
        for (auto [index, resource] : llvm::enumerate(limitedResources)) {
          assert(index < bindings->size() && "missing resource binding");
          bindingAttrs.push_back(DictionaryAttr::get(
              builder.getContext(),
              {NamedAttribute(builder.getStringAttr("resource"),
                              builder.getStringAttr(resource.getValue())),
               NamedAttribute(builder.getStringAttr("instance"),
                              builder.getI64IntegerAttr((*bindings)[index])),
               NamedAttribute(
                   builder.getStringAttr("hold"),
                   builder.getI64IntegerAttr(
                       problem.getResourceInitiationInterval(resource).value_or(
                           1))),
               NamedAttribute(builder.getStringAttr("period"),
                              builder.getI64IntegerAttr(
                                  *problem.getInitiationInterval()))}));
        }
        newOp->setAttr(resourceBindingsAttrName,
                       ArrayAttr::get(builder.getContext(), bindingAttrs));
      } else if (auto resources = problem.getLinkedResourceTypes(op)) {
        SmallVector<Attribute> reservations;
        for (auto resource : *resources) {
          auto tableIt = rotatingSelectorTables.find(resource);
          if (tableIt == rotatingSelectorTables.end())
            continue;
          auto &allocation = tableIt->second;
          auto selectorsIt = allocation.selectors.find(op);
          if (selectorsIt == allocation.selectors.end())
            continue;
          auto &selectors = selectorsIt->second;
          SmallVector<NamedAttribute> attributes{
              NamedAttribute(builder.getStringAttr("resource"),
                             builder.getStringAttr(resource.getValue())),
              NamedAttribute(builder.getStringAttr("phase"),
                             builder.getI64IntegerAttr(
                                 startTime % *problem.getInitiationInterval())),
              NamedAttribute(
                  builder.getStringAttr("period"),
                  builder.getI64IntegerAttr(*problem.getInitiationInterval())),
              NamedAttribute(
                  builder.getStringAttr("hold"),
                  builder.getI64IntegerAttr(
                      problem.getResourceInitiationInterval(resource).value_or(
                          1))),
              NamedAttribute(builder.getStringAttr("instances"),
                             builder.getI64IntegerAttr(allocation.instances))};
          SmallVector<Attribute> selectorAttrs;
          for (unsigned selector : selectors)
            selectorAttrs.push_back(builder.getI64IntegerAttr(selector));
          attributes.push_back(NamedAttribute(
              builder.getStringAttr("selector"),
              ArrayAttr::get(builder.getContext(), selectorAttrs)));
          attributes.push_back(
              NamedAttribute(builder.getStringAttr("selector_period"),
                             builder.getI64IntegerAttr(selectors.size())));
          reservations.push_back(
              DictionaryAttr::get(builder.getContext(), attributes));
        }
        if (!reservations.empty())
          newOp->setAttr(rotatingReservationsAttrName,
                         ArrayAttr::get(builder.getContext(), reservations));
      }

      // All further uses in this stage should used the cloned-version of values
      // So we update the mapping in this stage
      for (auto result : op->getResults())
        stageValueMaps[startTime].map(
            result, newOp->getResult(result.getResultNumber()));
    }

    // Register all values in the terminator, using their mapped value
    SmallVector<Value> stageOperands;
    unsigned resIndex = 0;
    for (auto res : registerValues[startTime]) {
      stageOperands.push_back(stageValueMaps[startTime].lookup(res));
      // Additionally, update the map of the stage that will consume the
      // registered value
      unsigned destTime = startTime + 1;
      unsigned latency = *problem.getLatency(
          *problem.getLinkedOperatorType(res.getDefiningOp()));
      // Multi-cycle case
      if (*problem.getStartTime(res.getDefiningOp()) == startTime &&
          latency > 1)
        destTime = startTime + latency;
      destTime = std::min((unsigned)(stageValueMaps.size() - 1), destTime);
      stageValueMaps[destTime].map(res, stage.getResult(resIndex++));
    }
    // Add these mapped values to pipeline.register
    stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                    stageOperands);

    // Add the induction variable increment to the first stage.
    if (startTime == 0) {
      auto incResult =
          arith::AddIOp::create(builder, stagesBlock.getArgument(0), step);
      stageTerminator->insertOperands(stageTerminator->getNumOperands(),
                                      incResult->getResults());
    }
  }

  // Add the iter args and results to the terminator.
  auto stagesTerminator =
      cast<LoopScheduleTerminatorOp>(stagesBlock.getTerminator());

  // Collect iter args and results from the induction variable increment and any
  // mapped values that were originally yielded.
  SmallVector<Value> termIterArgs;
  SmallVector<Value> termResults;
  termIterArgs.push_back(
      stagesBlock.front().getResult(stagesBlock.front().getNumResults() - 1));

  for (auto value : forOp.getBody()->getTerminator()->getOperands()) {
    unsigned lookupTime = std::min((unsigned)(stageValueMaps.size() - 1),
                                   pipeTimes[value.getDefiningOp()].second);

    termIterArgs.push_back(stageValueMaps[lookupTime].lookup(value));
    termResults.push_back(stageValueMaps[lookupTime].lookup(value));
  }

  stagesTerminator.getIterArgsMutable().append(termIterArgs);
  stagesTerminator.getResultsMutable().append(termResults);

  // Replace loop results with pipeline results.
  for (size_t i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(pipeline.getResult(i));

  // Remove the loop nest from the IR.
  loopNest.front().walk([](Operation *op) {
    op->dropAllUses();
    op->dropAllDefinedValueUses();
    op->dropAllReferences();
    op->erase();
  });

  return success();
}

std::unique_ptr<mlir::Pass> circt::createAffineToLoopSchedule() {
  return std::make_unique<AffineToLoopSchedule>();
}
