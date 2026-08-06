//===- NodeRLScheduler.cpp - Node-level reinforcement learning scheduler --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements resource-constrained acyclic and modulo schedulers whose priority
// policies are trained on each problem instance with REINFORCE. Acyclic actions
// select ready operations. Modulo actions order conflicting periodic resource
// reservations while an incremental difference-constraint solver computes
// start times. This keeps the learned component lightweight and commits only a
// verified feasible schedule.
//
//===----------------------------------------------------------------------===//

#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"

#include "ResourceSchedule.h"

#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <random>
#include <tuple>
#include <vector>

#define DEBUG_TYPE "node-rl-scheduler"

using namespace circt;
using namespace circt::scheduling;

namespace {

/// Features are deliberately cheap to compute.  They describe a node's
/// relation to the objective and its pressure on the target resources; the
/// final feature captures how long the node has remained ready.
enum NodeFeature : unsigned {
  Bias,
  ReachesObjective,
  CriticalPathHeight,
  TightMobility,
  Fanout,
  Latency,
  ResourcePressure,
  WaitingTime,
  NumFeatures
};

using FeatureVector = std::array<double, NumFeatures>;
using ResourceUsage =
    SmallDenseMap<Problem::ResourceType, SmallDenseMap<unsigned, unsigned>>;

struct NodeInfo {
  Operation *op = nullptr;
  SmallVector<unsigned> predecessors;
  SmallVector<unsigned> successors;
  SmallVector<Problem::ResourceType, 2> limitedResources;
  FeatureVector features{};
  unsigned latency = 0;
  unsigned asap = 0;
  unsigned height = 0;
  unsigned mobility = 0;
  bool reachesObjective = false;
};

struct EpisodeResult {
  SmallVector<unsigned> startTimes;
  FeatureVector policyGradient{};
  unsigned objectiveStart = 0;
  unsigned makespan = 0;
  uint64_t totalStartTime = 0;
  unsigned decisions = 0;
};

class NodeRLScheduler {
public:
  NodeRLScheduler(SharedOperatorsProblem &prob, Operation *lastOp,
                  const NodeRLSchedulerOptions &options)
      : prob(prob), lastOp(lastOp), options(options), rng(options.seed) {}

  LogicalResult schedule();

private:
  SharedOperatorsProblem &prob;
  Operation *lastOp;
  const NodeRLSchedulerOptions &options;
  std::mt19937_64 rng;

  SmallVector<NodeInfo> nodes;
  DenseMap<Operation *, unsigned> nodeIndices;
  SmallVector<unsigned> topologicalOrder;
  unsigned lastNode = 0;
  unsigned rewardScale = 1;

  LogicalResult initialize();
  EpisodeResult runEpisode(const FeatureVector &weights, bool stochastic,
                           double temperature);
  unsigned selectNode(ArrayRef<unsigned> candidates,
                      ArrayRef<unsigned> readyTimes, unsigned currentTime,
                      const FeatureVector &weights, bool stochastic,
                      double temperature, EpisodeResult &result);
  FeatureVector getFeatures(unsigned node, unsigned readyTime,
                            unsigned currentTime) const;
  bool resourcesAvailable(unsigned node, unsigned currentTime,
                          const ResourceUsage &usage) const;
  void reserveResources(unsigned node, unsigned currentTime,
                        ResourceUsage &usage) const;
  double getReward(const EpisodeResult &result) const;
  LogicalResult bindResources(const EpisodeResult &result);

  static bool isBetter(const EpisodeResult &candidate,
                       const EpisodeResult &best);
};

LogicalResult NodeRLScheduler::initialize() {
  if (!prob.hasOperation(lastOp))
    return prob.getContainingOp()->emitError(
        "problem does not include last operation");

  for (auto *op : prob.getOperations()) {
    unsigned node = nodes.size();
    nodeIndices[op] = node;
    NodeInfo info;
    info.op = op;
    info.latency = *prob.getLatency(*prob.getLinkedOperatorType(op));
    nodes.push_back(std::move(info));
  }
  lastNode = nodeIndices.lookup(lastOp);

  // Materialize the predecessor/successor relation once.  Multiple SSA uses
  // between the same pair of operations represent one precedence relation for
  // the purposes of list scheduling.
  for (unsigned dst = 0, e = nodes.size(); dst != e; ++dst) {
    llvm::SmallDenseSet<unsigned, 4> seenPredecessors;
    for (auto dep : prob.getDependences(nodes[dst].op)) {
      auto sourceIt = nodeIndices.find(dep.getSource());
      if (sourceIt == nodeIndices.end())
        return prob.getContainingOp()->emitError(
            "dependence source is not part of the scheduling problem");
      unsigned src = sourceIt->second;
      if (!seenPredecessors.insert(src).second)
        continue;
      nodes[dst].predecessors.push_back(src);
      nodes[src].successors.push_back(dst);
    }
  }

  // Obtain a stable topological order and diagnose cyclic inputs before any
  // training episodes are run.
  DenseSet<Operation *> handled;
  if (failed(handleOperationsInTopologicalOrder(
          prob, [&](Operation *op) -> LogicalResult {
            unsigned node = nodeIndices.lookup(op);
            if (llvm::any_of(nodes[node].predecessors, [&](unsigned pred) {
                  return !handled.contains(nodes[pred].op);
                }))
              return failure();
            handled.insert(op);
            topologicalOrder.push_back(node);
            return success();
          })))
    return failure();

  // Compute resource-free ASAP times.
  for (unsigned node : topologicalOrder) {
    for (unsigned pred : nodes[node].predecessors)
      nodes[node].asap =
          std::max(nodes[node].asap, nodes[pred].asap + nodes[pred].latency);
  }
  rewardScale = std::max(1u, nodes[lastNode].asap);

  // Mark the transitive fan-in of the objective and compute each node's
  // latency-weighted distance to it.
  nodes[lastNode].reachesObjective = true;
  for (unsigned node : llvm::reverse(topologicalOrder)) {
    if (node == lastNode)
      continue;
    for (unsigned succ : nodes[node].successors) {
      if (!nodes[succ].reachesObjective)
        continue;
      nodes[node].reachesObjective = true;
      nodes[node].height = std::max(nodes[node].height,
                                    nodes[node].latency + nodes[succ].height);
    }
  }

  // Compute ALAP mobility against the resource-free objective time.  Mobility
  // is only meaningful for nodes that can reach the requested objective.
  SmallVector<unsigned> latest(nodes.size(),
                               std::numeric_limits<unsigned>::max());
  latest[lastNode] = nodes[lastNode].asap;
  for (unsigned node : llvm::reverse(topologicalOrder)) {
    if (node == lastNode || !nodes[node].reachesObjective)
      continue;
    for (unsigned succ : nodes[node].successors) {
      if (!nodes[succ].reachesObjective ||
          latest[succ] == std::numeric_limits<unsigned>::max())
        continue;
      assert(latest[succ] >= nodes[node].latency &&
             "resource-free objective must admit an ALAP schedule");
      latest[node] = std::min(latest[node], latest[succ] - nodes[node].latency);
    }
    assert(latest[node] != std::numeric_limits<unsigned>::max());
    nodes[node].mobility = latest[node] - nodes[node].asap;
  }

  // Record the limited resources used by each node and estimate their global
  // pressure.  Unlike the simplex heuristic, the environment naturally
  // supports operations that require more than one resource type.
  DenseMap<Problem::ResourceType, unsigned> resourceDemand;
  for (auto &node : nodes) {
    auto resources = prob.getLinkedResourceTypes(node.op);
    if (!resources)
      continue;
    for (auto resource : *resources) {
      if (prob.getLimit(resource).value_or(0) == 0 ||
          llvm::is_contained(node.limitedResources, resource))
        continue;
      node.limitedResources.push_back(resource);
      ++resourceDemand[resource];
    }
  }

  unsigned maxHeight = 0;
  unsigned maxMobility = 0;
  unsigned maxFanout = 0;
  unsigned maxLatency = 0;
  double maxPressure = 0.0;
  SmallVector<double> pressures(nodes.size(), 0.0);
  for (auto [index, node] : llvm::enumerate(nodes)) {
    maxHeight = std::max(maxHeight, node.height);
    maxMobility = std::max(maxMobility, node.mobility);
    maxFanout = std::max<unsigned>(maxFanout, node.successors.size());
    maxLatency = std::max(maxLatency, node.latency);
    for (auto resource : node.limitedResources) {
      double pressure =
          static_cast<double>(resourceDemand.lookup(resource)) /
          *prob.getLimit(resource) *
          prob.getResourceInitiationInterval(resource).value_or(1);
      pressures[index] = std::max(pressures[index], pressure);
    }
    maxPressure = std::max(maxPressure, pressures[index]);
  }

  // Normalize all static features to [0, 1].  This keeps the policy update
  // well-conditioned across graphs of very different sizes and latencies.
  for (auto [index, node] : llvm::enumerate(nodes)) {
    node.features[Bias] = 1.0;
    node.features[ReachesObjective] = node.reachesObjective ? 1.0 : 0.0;
    node.features[CriticalPathHeight] =
        maxHeight == 0 ? 0.0 : static_cast<double>(node.height) / maxHeight;
    node.features[TightMobility] =
        node.reachesObjective ? 1.0 - static_cast<double>(node.mobility) /
                                          std::max(1u, maxMobility)
                              : 0.0;
    node.features[Fanout] =
        maxFanout == 0
            ? 0.0
            : static_cast<double>(node.successors.size()) / maxFanout;
    node.features[Latency] =
        maxLatency == 0 ? 0.0 : static_cast<double>(node.latency) / maxLatency;
    node.features[ResourcePressure] =
        maxPressure == 0.0 ? 0.0 : pressures[index] / maxPressure;
  }

  return success();
}

FeatureVector NodeRLScheduler::getFeatures(unsigned node, unsigned readyTime,
                                           unsigned currentTime) const {
  FeatureVector features = nodes[node].features;
  features[WaitingTime] =
      std::min(1.0, static_cast<double>(currentTime - readyTime) / rewardScale);
  return features;
}

bool NodeRLScheduler::resourcesAvailable(unsigned node, unsigned currentTime,
                                         const ResourceUsage &usage) const {
  return llvm::all_of(nodes[node].limitedResources, [&](auto resource) {
    unsigned ii = prob.getResourceInitiationInterval(resource).value_or(1);
    for (unsigned time = currentTime, end = currentTime + ii; time != end;
         ++time)
      if (usage.lookup(resource).lookup(time) >= *prob.getLimit(resource))
        return false;
    return true;
  });
}

void NodeRLScheduler::reserveResources(unsigned node, unsigned currentTime,
                                       ResourceUsage &usage) const {
  for (auto resource : nodes[node].limitedResources) {
    unsigned ii = prob.getResourceInitiationInterval(resource).value_or(1);
    for (unsigned time = currentTime, end = currentTime + ii; time != end;
         ++time)
      ++usage[resource][time];
  }
}

unsigned NodeRLScheduler::selectNode(ArrayRef<unsigned> candidates,
                                     ArrayRef<unsigned> readyTimes,
                                     unsigned currentTime,
                                     const FeatureVector &weights,
                                     bool stochastic, double temperature,
                                     EpisodeResult &result) {
  SmallVector<FeatureVector> candidateFeatures;
  SmallVector<double> scores;
  candidateFeatures.reserve(candidates.size());
  scores.reserve(candidates.size());

  for (unsigned node : candidates) {
    auto features = getFeatures(node, readyTimes[node], currentTime);
    double score = 0.0;
    for (unsigned feature = 0; feature != NumFeatures; ++feature)
      score += weights[feature] * features[feature];
    candidateFeatures.push_back(features);
    scores.push_back(score);
  }

  if (!stochastic || candidates.size() == 1) {
    unsigned best = 0;
    for (unsigned i = 1, e = scores.size(); i != e; ++i)
      if (scores[i] > scores[best])
        best = i;
    return best;
  }

  // Sample an action from a numerically stable softmax distribution.
  temperature = std::max(temperature, 1.0e-6);
  double maxScore = *std::max_element(scores.begin(), scores.end());
  SmallVector<double> probabilities;
  probabilities.reserve(scores.size());
  double probabilitySum = 0.0;
  for (double score : scores) {
    double probability = std::exp((score - maxScore) / temperature);
    probabilities.push_back(probability);
    probabilitySum += probability;
  }
  for (double &probability : probabilities)
    probability /= probabilitySum;

  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double sample = distribution(rng);
  unsigned selected = probabilities.size() - 1;
  double cumulative = 0.0;
  for (unsigned i = 0, e = probabilities.size(); i != e; ++i) {
    cumulative += probabilities[i];
    if (sample < cumulative) {
      selected = i;
      break;
    }
  }

  // Gradient of log softmax(a | s): f(s,a) - E[f(s,.)].
  for (unsigned feature = 0; feature != NumFeatures; ++feature) {
    double expected = 0.0;
    for (unsigned i = 0, e = candidates.size(); i != e; ++i)
      expected += probabilities[i] * candidateFeatures[i][feature];
    result.policyGradient[feature] +=
        (candidateFeatures[selected][feature] - expected) / temperature;
  }
  ++result.decisions;
  return selected;
}

EpisodeResult NodeRLScheduler::runEpisode(const FeatureVector &weights,
                                          bool stochastic, double temperature) {
  EpisodeResult result;
  result.startTimes.resize(nodes.size());
  SmallVector<char> scheduled(nodes.size(), false);
  SmallVector<unsigned> readyTimes(nodes.size());
  unsigned numScheduled = 0;
  unsigned currentTime = 0;
  ResourceUsage usage;

  while (numScheduled != nodes.size()) {
    // Keep scheduling at this time step.  Zero-latency nodes can make further
    // nodes ready in the same cycle, so readiness is recomputed after each
    // action.
    while (true) {
      SmallVector<unsigned> candidates;
      for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
        if (scheduled[node])
          continue;

        bool allPredecessorsScheduled = true;
        unsigned readyTime = 0;
        for (unsigned pred : nodes[node].predecessors) {
          if (!scheduled[pred]) {
            allPredecessorsScheduled = false;
            break;
          }
          readyTime = std::max(readyTime,
                               result.startTimes[pred] + nodes[pred].latency);
        }
        if (!allPredecessorsScheduled)
          continue;

        readyTimes[node] = readyTime;
        if (readyTime <= currentTime &&
            resourcesAvailable(node, currentTime, usage))
          candidates.push_back(node);
      }

      if (candidates.empty())
        break;

      unsigned selected = selectNode(candidates, readyTimes, currentTime,
                                     weights, stochastic, temperature, result);
      unsigned node = candidates[selected];
      result.startTimes[node] = currentTime;
      scheduled[node] = true;
      ++numScheduled;
      reserveResources(node, currentTime, usage);
    }

    if (numScheduled == nodes.size())
      break;

    // Skip empty cycles when all currently precedence-ready nodes have a
    // future release time.  A node blocked by this cycle's resource usage has
    // a ready time <= currentTime and therefore advances us by exactly one.
    unsigned nextReady = std::numeric_limits<unsigned>::max();
    for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
      if (scheduled[node])
        continue;
      bool allPredecessorsScheduled = true;
      unsigned readyTime = 0;
      for (unsigned pred : nodes[node].predecessors) {
        if (!scheduled[pred]) {
          allPredecessorsScheduled = false;
          break;
        }
        readyTime =
            std::max(readyTime, result.startTimes[pred] + nodes[pred].latency);
      }
      if (allPredecessorsScheduled)
        nextReady = std::min(nextReady, readyTime);
    }
    assert(nextReady != std::numeric_limits<unsigned>::max() &&
           "acyclic graph must have a precedence-ready node");
    currentTime = std::max(currentTime + 1, nextReady);
  }

  result.objectiveStart = result.startTimes[lastNode];
  for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
    result.makespan = std::max(result.makespan,
                               result.startTimes[node] + nodes[node].latency);
    result.totalStartTime += result.startTimes[node];
  }
  return result;
}

double NodeRLScheduler::getReward(const EpisodeResult &result) const {
  // The small secondary term gives the learner a signal between schedules with
  // equal objective time without changing the scheduler's primary objective.
  double objective = static_cast<double>(result.objectiveStart) / rewardScale;
  double secondary = static_cast<double>(result.totalStartTime) /
                     (std::max<size_t>(1, nodes.size()) * rewardScale);
  return -objective - 1.0e-3 * secondary;
}

bool NodeRLScheduler::isBetter(const EpisodeResult &candidate,
                               const EpisodeResult &best) {
  if (candidate.objectiveStart != best.objectiveStart)
    return candidate.objectiveStart < best.objectiveStart;
  if (candidate.makespan != best.makespan)
    return candidate.makespan < best.makespan;
  if (candidate.totalStartTime != best.totalStartTime)
    return candidate.totalStartTime < best.totalStartTime;
  return std::lexicographical_compare(
      candidate.startTimes.begin(), candidate.startTimes.end(),
      best.startTimes.begin(), best.startTimes.end());
}

LogicalResult NodeRLScheduler::bindResources(const EpisodeResult &result) {
  struct ResourceUse {
    unsigned startTime;
    unsigned node;
    unsigned resourceIndex;
  };

  DenseMap<Problem::ResourceType, SmallVector<ResourceUse, 4>> uses;
  SmallVector<SmallVector<unsigned, 2>> bindings(nodes.size());
  for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
    bindings[node].resize(nodes[node].limitedResources.size());
    for (auto [resourceIndex, resource] :
         llvm::enumerate(nodes[node].limitedResources))
      uses[resource].push_back({result.startTimes[node], node,
                                static_cast<unsigned>(resourceIndex)});
  }

  for (auto &[resource, resourceUses] : uses) {
    llvm::sort(resourceUses,
               [](const ResourceUse &lhs, const ResourceUse &rhs) {
                 return std::tie(lhs.startTime, lhs.node) <
                        std::tie(rhs.startTime, rhs.node);
               });
    SmallVector<unsigned> nextAvailable(*prob.getLimit(resource), 0);
    unsigned ii = prob.getResourceInitiationInterval(resource).value_or(1);
    for (const auto &use : resourceUses) {
      auto instance = llvm::find_if(nextAvailable, [&](unsigned available) {
        return available <= use.startTime;
      });
      if (instance == nextAvailable.end())
        return prob.getContainingOp()->emitError()
               << "could not bind resource '" << resource.getValue() << "'";
      unsigned instanceIndex = instance - nextAvailable.begin();
      *instance = use.startTime + ii;
      bindings[use.node][use.resourceIndex] = instanceIndex;
    }
  }

  prob.clearResourceBindings();
  for (unsigned node = 0, e = nodes.size(); node != e; ++node)
    if (!bindings[node].empty())
      prob.setResourceBindings(nodes[node].op, std::move(bindings[node]));
  return success();
}

LogicalResult NodeRLScheduler::schedule() {
  if (options.episodes == 0)
    return prob.getContainingOp()->emitError(
        "node-rl requires at least one training episode");
  if (!std::isfinite(options.learningRate) || options.learningRate <= 0.0)
    return prob.getContainingOp()->emitError(
        "node-rl learning rate must be finite and positive");
  if (!std::isfinite(options.exploration) || options.exploration <= 0.0)
    return prob.getContainingOp()->emitError(
        "node-rl exploration must be finite and positive");
  prob.clearResourceBindings();
  if (failed(initialize()))
    return failure();

  // An insertion-order list schedule is a conservative fallback.  Keeping the
  // best result across all episodes means exploration can never make the
  // returned schedule worse than this baseline.
  FeatureVector zeroWeights{};
  EpisodeResult best = runEpisode(zeroWeights, false, 1.0);

  // Static graph features provide a useful initial policy before the first
  // reward is observed.  Training adapts these weights to the concrete graph.
  // Criticality is valuable for latency refinement on compact graphs. On a
  // large low-capacity resource cluster, however, prioritizing the objective
  // fan-in too early can spend the global work budget on infeasible orders
  // before reaching the resource-bound II. Start those cases with the robust
  // resource-pressure policy and leave objective refinement to subsequent
  // learning or the optional bounded local search.
  FeatureVector weights =
      nodes.size() < 512
          ? FeatureVector{0.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.25}
          : FeatureVector{0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0};
  EpisodeResult initialPolicy = runEpisode(weights, false, 1.0);
  if (isBetter(initialPolicy, best))
    best = initialPolicy;

  double averageReward = getReward(initialPolicy);
  for (unsigned episode = 0; episode != options.episodes; ++episode) {
    double progress = options.episodes == 1 ? 1.0
                                            : static_cast<double>(episode) /
                                                  (options.episodes - 1);
    double temperature =
        std::max(0.05, options.exploration * std::pow(0.1, progress));
    EpisodeResult sample = runEpisode(weights, true, temperature);
    if (isBetter(sample, best))
      best = sample;

    double reward = getReward(sample);
    double advantage = reward - averageReward;
    if (sample.decisions != 0) {
      double step = options.learningRate * advantage / sample.decisions;
      for (unsigned feature = 0; feature != NumFeatures; ++feature)
        weights[feature] =
            std::clamp(weights[feature] + step * sample.policyGradient[feature],
                       -12.0, 12.0);
    }
    averageReward = 0.9 * averageReward + 0.1 * reward;

    // Periodically evaluate the greedy policy because the sampled trajectory
    // may not itself expose the improvement represented by the updated weights.
    if ((episode + 1) % 8 == 0 || episode + 1 == options.episodes) {
      EpisodeResult greedy = runEpisode(weights, false, 1.0);
      if (isBetter(greedy, best))
        best = std::move(greedy);
    }
  }

  for (unsigned node = 0, e = nodes.size(); node != e; ++node)
    prob.setStartTime(nodes[node].op, best.startTimes[node]);
  if (failed(bindResources(best)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "node-rl selected objective t="
                          << best.objectiveStart << " after "
                          << options.episodes << " episodes\n");
  return success();
}

/// A modulo schedule is represented by ordinary difference constraints. A
/// dependence `src -> dst` at distance d becomes
/// `start(dst) >= start(src) + latency(src) - d * II`. Resource conflicts add
/// the same kind of constraints, but only after a policy has chosen an order
/// for the conflicting nodes. This makes every policy episode a small, fully
/// checked constraint-solving problem instead of requiring a neural model to
/// predict absolute time steps directly.
struct ModuloConstraint {
  unsigned source;
  unsigned destination;
  unsigned delay;
  // Dependence distances are nonnegative. Resource-ordering constraints may
  // use a negative distance to distinguish operations whose absolute start
  // times differ by one or more IIs but whose modulo phases still collide.
  int64_t distance;
};

using ModuloConstraintKey = std::tuple<unsigned, unsigned, unsigned, int64_t>;

static ModuloConstraintKey
getConstraintKey(const ModuloConstraint &constraint) {
  return {constraint.source, constraint.destination, constraint.delay,
          constraint.distance};
}

struct ClusterResource {
  Problem::ResourceType resource;
  SmallVector<unsigned> nodes;
};

/// A maximal scheduling cluster after joining all users of a limited resource
/// and all strongly connected components of the resulting dependence graph.
/// Constraints inside a cluster must be solved together. Constraints between
/// clusters form a DAG and can be satisfied by shifting whole clusters without
/// changing any modulo-resource conflicts.
struct ModuloCluster {
  SmallVector<unsigned> nodes;
  SmallVector<ModuloConstraint> dependenceConstraints;
  SmallVector<ClusterResource> resources;
};

struct ResourceConflict {
  unsigned source;
  unsigned destination;
  unsigned sourcePhase;
  unsigned destinationPhase;
  unsigned resourceII;
};

/// Incrementally maintains modulo-reservation occupancy for a cluster. Heap
/// entries are invalidated with a per-node generation when propagation moves a
/// node to another phase. This avoids rebuilding every resource's complete
/// reservation table after each newly added ordering constraint.
class ClusterReservationState {
public:
  ClusterReservationState(const ModuloCluster &cluster,
                          ArrayRef<int64_t> startTimes, unsigned ii,
                          ModuloProblem &prob)
      : ii(ii), nodePhases(cluster.nodes.size()),
        nodeGenerations(cluster.nodes.size(), 1),
        resourcesByNode(cluster.nodes.size()) {
    tables.reserve(cluster.resources.size());
    for (auto [resourceIndex, resourceUse] :
         llvm::enumerate(cluster.resources)) {
      unsigned limit = *prob.getLimit(resourceUse.resource);
      unsigned resourceII =
          prob.getResourceInitiationInterval(resourceUse.resource).value_or(1);
      tables.emplace_back(limit, resourceII, ii);
      for (unsigned node : resourceUse.nodes)
        resourcesByNode[node].push_back(resourceIndex);
    }

    for (unsigned node = 0, e = cluster.nodes.size(); node != e; ++node) {
      assert(startTimes[node] >= 0 &&
             "difference solution must be nonnegative");
      unsigned phase = static_cast<uint64_t>(startTimes[node]) % ii;
      nodePhases[node] = phase;
      for (unsigned resourceIndex : resourcesByNode[node])
        tables[resourceIndex].add(node, phase, nodeGenerations[node], ii);
    }
  }

  std::optional<ResourceConflict> findConflict() {
    for (auto &table : tables) {
      for (unsigned phase = 0; phase != ii; ++phase) {
        if (table.occupancy[phase] <= table.limit)
          continue;
        auto &reservations = table.reservations[phase];
        discardStale(reservations);
        assert(!reservations.empty() && "overfull phase must have a user");
        ReservationEntry first = reservations.top();
        reservations.pop();
        discardStale(reservations);
        assert(!reservations.empty() && "overfull phase must have two users");
        ReservationEntry second = reservations.top();
        reservations.push(first);
        unsigned source = std::get<1>(first);
        unsigned destination = std::get<1>(second);
        return ResourceConflict{source, destination, nodePhases[source],
                                nodePhases[destination], table.resourceII};
      }
    }
    return std::nullopt;
  }

  void update(ArrayRef<unsigned> changedNodes, ArrayRef<int64_t> startTimes) {
    for (unsigned node : changedNodes) {
      assert(startTimes[node] >= 0 &&
             "difference solution must be nonnegative");
      unsigned newPhase = static_cast<uint64_t>(startTimes[node]) % ii;
      unsigned oldPhase = nodePhases[node];
      if (newPhase == oldPhase)
        continue;
      ++nodeGenerations[node];
      for (unsigned resourceIndex : resourcesByNode[node]) {
        auto &table = tables[resourceIndex];
        table.remove(oldPhase, ii);
        table.add(node, newPhase, nodeGenerations[node], ii);
      }
      nodePhases[node] = newPhase;
    }
  }

private:
  using ReservationEntry = std::tuple<unsigned, unsigned, uint64_t>;
  using ReservationQueue =
      std::priority_queue<ReservationEntry, std::vector<ReservationEntry>,
                          std::greater<ReservationEntry>>;

  struct ResourceTable {
    ResourceTable(unsigned limit, unsigned resourceII, unsigned ii)
        : limit(limit), resourceII(resourceII), occupancy(ii, 0),
          reservations(ii) {}

    void add(unsigned node, unsigned phase, uint64_t generation, unsigned ii) {
      for (unsigned offset = 0; offset != resourceII; ++offset) {
        unsigned activePhase = (static_cast<uint64_t>(phase) + offset) % ii;
        ++occupancy[activePhase];
        reservations[activePhase].push({phase, node, generation});
      }
    }

    void remove(unsigned phase, unsigned ii) {
      for (unsigned offset = 0; offset != resourceII; ++offset) {
        unsigned activePhase = (static_cast<uint64_t>(phase) + offset) % ii;
        assert(occupancy[activePhase] != 0 && "removing absent reservation");
        --occupancy[activePhase];
      }
    }

    unsigned limit;
    unsigned resourceII;
    SmallVector<unsigned> occupancy;
    SmallVector<ReservationQueue> reservations;
  };

  void discardStale(ReservationQueue &reservations) const {
    while (!reservations.empty() &&
           std::get<2>(reservations.top()) !=
               nodeGenerations[std::get<1>(reservations.top())])
      reservations.pop();
  }

  unsigned ii;
  SmallVector<unsigned> nodePhases;
  SmallVector<uint64_t> nodeGenerations;
  SmallVector<SmallVector<unsigned, 2>> resourcesByNode;
  SmallVector<ResourceTable> tables;
};

class DisjointSet {
public:
  explicit DisjointSet(unsigned size) : parents(size), ranks(size, 0) {
    for (unsigned index = 0; index != size; ++index)
      parents[index] = index;
  }

  unsigned find(unsigned value) {
    unsigned root = value;
    while (parents[root] != root)
      root = parents[root];
    while (parents[value] != value) {
      unsigned parent = parents[value];
      parents[value] = root;
      value = parent;
    }
    return root;
  }

  void unite(unsigned lhs, unsigned rhs) {
    lhs = find(lhs);
    rhs = find(rhs);
    if (lhs == rhs)
      return;
    if (ranks[lhs] < ranks[rhs])
      std::swap(lhs, rhs);
    parents[rhs] = lhs;
    if (ranks[lhs] == ranks[rhs])
      ++ranks[lhs];
  }

private:
  SmallVector<unsigned> parents;
  SmallVector<unsigned> ranks;
};

/// Incrementally maintains the least solution of a feasible set of difference
/// constraints. The base solution already satisfies `baseConstraints`; every
/// subsequently added resource-ordering edge only increases affected start
/// times, so a worklist can propagate that change without resolving the full
/// graph from zero.
class IncrementalDifferenceSolver {
public:
  IncrementalDifferenceSolver(unsigned numNodes, unsigned ii,
                              ArrayRef<ModuloConstraint> baseConstraints,
                              ArrayRef<int64_t> baseStartTimes)
      : ii(ii), constraints(baseConstraints.begin(), baseConstraints.end()),
        outgoing(numNodes), queuedGeneration(numNodes, 0),
        changedGeneration(numNodes, 0),
        startTimes(baseStartTimes.begin(), baseStartTimes.end()) {
    assert(startTimes.size() == numNodes && "invalid base solution size");
    for (auto [index, constraint] : llvm::enumerate(constraints))
      outgoing[constraint.source].push_back(index);
  }

  /// Add one edge to the currently feasible graph. Since the old graph has no
  /// positive cycle, any positive cycle introduced here must contain the new
  /// edge. Re-relaxing that edge after its initial application therefore
  /// detects infeasibility exactly.
  bool addConstraint(const ModuloConstraint &constraint,
                     SmallVectorImpl<unsigned> *changedNodes = nullptr) {
    if (changedNodes)
      changedNodes->clear();
    unsigned newConstraint = constraints.size();
    constraints.push_back(constraint);
    outgoing[constraint.source].push_back(newConstraint);

    int64_t required = getRequiredStart(constraint);
    if (required <= startTimes[constraint.destination])
      return true;
    startTimes[constraint.destination] = required;

    worklist.clear();
    worklist.push_back(constraint.destination);
    if (++generation == 0) {
      llvm::fill(queuedGeneration, 0);
      llvm::fill(changedGeneration, 0);
      generation = 1;
    }
    auto recordChange = [&](unsigned node) {
      if (!changedNodes || changedGeneration[node] == generation)
        return;
      changedGeneration[node] = generation;
      changedNodes->push_back(node);
    };
    recordChange(constraint.destination);
    queuedGeneration[constraint.destination] = generation;
    for (size_t cursor = 0; cursor != worklist.size(); ++cursor) {
      unsigned source = worklist[cursor];
      queuedGeneration[source] = 0;
      for (unsigned constraintIndex : outgoing[source]) {
        const auto &outgoingConstraint = constraints[constraintIndex];
        int64_t outgoingRequired = getRequiredStart(outgoingConstraint);
        if (outgoingRequired <= startTimes[outgoingConstraint.destination])
          continue;
        if (constraintIndex == newConstraint)
          return false;
        startTimes[outgoingConstraint.destination] = outgoingRequired;
        recordChange(outgoingConstraint.destination);
        if (queuedGeneration[outgoingConstraint.destination] != generation) {
          queuedGeneration[outgoingConstraint.destination] = generation;
          worklist.push_back(outgoingConstraint.destination);
        }
      }
    }
    return true;
  }

  ArrayRef<int64_t> getStartTimes() const { return startTimes; }

private:
  int64_t getRequiredStart(const ModuloConstraint &constraint) const {
    return startTimes[constraint.source] + constraint.delay -
           static_cast<int64_t>(constraint.distance) * ii;
  }

  unsigned ii;
  SmallVector<ModuloConstraint> constraints;
  SmallVector<SmallVector<unsigned, 4>> outgoing;
  SmallVector<unsigned> queuedGeneration;
  SmallVector<unsigned> changedGeneration;
  SmallVector<unsigned> worklist;
  unsigned generation = 0;
  SmallVector<int64_t> startTimes;
};

/// A copyable prefix in the fixed-II resource-ordering search. Every state is
/// already dependence-feasible; adding one action either produces another
/// exact difference-constraint solution or proves that branch infeasible.
struct ModuloSearchState {
  ModuloSearchState(const ModuloCluster &cluster, unsigned ii,
                    ArrayRef<int64_t> baseStartTimes)
      : solver(cluster.nodes.size(), ii, cluster.dependenceConstraints,
               baseStartTimes) {
    for (const auto &constraint : cluster.dependenceConstraints)
      constraintKeys.push_back(getConstraintKey(constraint));
  }

  IncrementalDifferenceSolver solver;
  SmallVector<ModuloConstraintKey, 16> constraintKeys;
};

struct MCTSActionCandidate {
  ModuloConstraint action;
  uint64_t resourceExcess;
};

struct RankedResourceConflict {
  ResourceConflict conflict;
  unsigned overload;
  unsigned resourceIndex;
  unsigned activePhase;
};

struct MCTSTreeNode {
  explicit MCTSTreeNode(ModuloSearchState state, unsigned depth = 0)
      : state(std::move(state)), depth(depth) {}

  ModuloSearchState state;
  SmallVector<ModuloConstraint, 16> actions;
  SmallVector<unsigned, 16> children;
  unsigned nextAction = 0;
  unsigned visits = 0;
  double totalReward = 0.0;
  unsigned depth = 0;
  bool actionsReady = false;
  bool feasible = false;
  bool deadEnd = false;
};

struct MCTSTree {
  MCTSTree(const ModuloSearchState &root, uint64_t seed) : rng(seed) {
    nodes.emplace_back(root);
  }

  SmallVector<MCTSTreeNode, 8> nodes;
  std::mt19937_64 rng;
};

struct ModuloClusterEpisode {
  SmallVector<int64_t> startTimes;
  FeatureVector policyGradient{};
  unsigned decisions = 0;
  bool feasible = false;
};

struct ModuloEpisode {
  SmallVector<int64_t> startTimes;
  FeatureVector policyGradient{};
  unsigned objectiveStart = 0;
  unsigned decisions = 0;
  bool feasible = false;
};

class ModuloNodeRLScheduler {
public:
  ModuloNodeRLScheduler(ModuloProblem &prob, Operation *lastOp,
                        const NodeRLSchedulerOptions &options)
      : prob(prob), lastOp(lastOp), options(options), rng(options.seed) {}

  LogicalResult schedule();

private:
  // Backtracking copies an incremental solver at each branch, so keep it as a
  // bounded repair for small ambiguous clusters. Larger clusters retain the
  // linear-memory policy path.
  static constexpr unsigned maxBacktrackingClusterSize = 32;
  static constexpr unsigned maxBacktrackingStates = 16384;
  static constexpr unsigned maxBacktrackingLiveStates = 64;

  ModuloProblem &prob;
  Operation *lastOp;
  const NodeRLSchedulerOptions &options;
  std::mt19937_64 rng;
  SmallVector<NodeInfo> nodes;
  DenseMap<Operation *, unsigned> nodeIndices;
  SmallVector<ModuloConstraint> dependenceConstraints;
  SmallVector<ModuloCluster> clusters;
  SmallVector<unsigned> clusterOfNode;
  SmallVector<unsigned> localIndexOfNode;
  SmallVector<SmallVector<ModuloConstraint, 4>> outgoingClusterConstraints;
  SmallVector<unsigned> clusterTopologicalOrder;
  unsigned lastNode = 0;
  uint64_t resourceOrderings = 0;
  bool resourceOrderingBudgetExhausted = false;

  LogicalResult initialize();
  LogicalResult buildClusters();
  bool solveDifferenceConstraints(ArrayRef<ModuloConstraint> constraints,
                                  unsigned numNodes, unsigned ii,
                                  SmallVectorImpl<int64_t> &startTimes) const;
  std::optional<ModuloConstraint> chooseResourceConstraint(
      const ModuloCluster &cluster, ArrayRef<int64_t> startTimes, unsigned ii,
      const FeatureVector &weights, bool stochastic, double temperature,
      bool wrapAround, ModuloClusterEpisode &episode);
  ModuloConstraint
  orderResourceConflict(const ModuloCluster &cluster, ResourceConflict conflict,
                        ArrayRef<int64_t> startTimes, unsigned ii,
                        const FeatureVector &weights, bool stochastic,
                        double temperature, bool wrapAround,
                        ModuloClusterEpisode &episode);
  bool shouldTrackReservations(const ModuloCluster &cluster) const;
  bool bindResources(ArrayRef<int64_t> startTimes, unsigned ii);
  ModuloClusterEpisode runClusterEpisode(const ModuloCluster &cluster,
                                         unsigned ii,
                                         ArrayRef<int64_t> baseStartTimes,
                                         const FeatureVector &weights,
                                         bool stochastic, double temperature,
                                         bool wrapAround, bool backtracking);
  ModuloClusterEpisode
  tryBacktrackingResourceOrdering(const ModuloCluster &cluster, unsigned ii,
                                  ArrayRef<int64_t> baseStartTimes,
                                  const FeatureVector &weights);
  ModuloClusterEpisode tryBulkResourceOrdering(const ModuloCluster &cluster,
                                               unsigned ii,
                                               ArrayRef<int64_t> baseStartTimes,
                                               const FeatureVector &weights,
                                               bool stochastic,
                                               double temperature);
  uint64_t countResourceExcess(const ModuloCluster &cluster,
                               ArrayRef<int64_t> startTimes, unsigned ii) const;
  ModuloConstraint
  orderResourceConflictAlternative(ResourceConflict conflict,
                                   ArrayRef<int64_t> startTimes, unsigned ii,
                                   bool alternative) const;
  bool applyMCTSAction(ModuloSearchState &state,
                       const ModuloConstraint &action);
  bool collectMCTSActions(const ModuloCluster &cluster,
                          const ModuloSearchState &state, unsigned ii,
                          unsigned conflictWidth, bool maskIllegal,
                          SmallVectorImpl<ModuloConstraint> &actions);
  void packTightUnitCapacityResources(
      const ModuloCluster &cluster, unsigned ii, ModuloSearchState &state,
      std::chrono::steady_clock::time_point deadline);
  ModuloClusterEpisode
  tryMCTSResourceOrdering(const ModuloCluster &cluster, unsigned ii,
                          ArrayRef<int64_t> baseStartTimes,
                          std::chrono::steady_clock::time_point deadline);
  ModuloEpisode
  runMCTSRescue(unsigned ii,
                ArrayRef<SmallVector<int64_t>> clusterBaseStartTimes,
                std::chrono::steady_clock::time_point deadline);
  ModuloEpisode runEpisode(unsigned ii,
                           ArrayRef<SmallVector<int64_t>> clusterBaseStartTimes,
                           const FeatureVector &weights, bool stochastic,
                           double temperature, bool wrapAround,
                           bool backtracking);
  bool solveClusterDependences(
      unsigned ii,
      SmallVectorImpl<SmallVector<int64_t>> &clusterStartTimes) const;
  void combineClusterSchedules(unsigned ii,
                               ArrayRef<SmallVector<int64_t>> clusterStartTimes,
                               ModuloEpisode &episode) const;
  unsigned computeLowerBound() const;
  unsigned computeEpisodePatience() const;
  bool consumeResourceOrdering();
};

LogicalResult ModuloNodeRLScheduler::initialize() {
  if (!prob.hasOperation(lastOp))
    return prob.getContainingOp()->emitError(
        "problem does not include last operation");

  for (auto *op : prob.getOperations()) {
    unsigned index = nodes.size();
    nodeIndices[op] = index;
    NodeInfo info;
    info.op = op;
    info.latency = *prob.getLatency(*prob.getLinkedOperatorType(op));
    nodes.push_back(std::move(info));
  }
  lastNode = nodeIndices.lookup(lastOp);

  DenseMap<Problem::ResourceType, unsigned> demand;
  for (unsigned dst = 0, e = nodes.size(); dst != e; ++dst) {
    for (auto dep : prob.getDependences(nodes[dst].op)) {
      auto source = nodeIndices.find(dep.getSource());
      if (source == nodeIndices.end())
        return prob.getContainingOp()->emitError(
            "dependence source is not part of the scheduling problem");
      nodes[dst].predecessors.push_back(source->second);
      nodes[source->second].successors.push_back(dst);
      dependenceConstraints.push_back(
          {source->second, dst, nodes[source->second].latency,
           static_cast<int64_t>(prob.getDistance(dep).value_or(0))});
    }
    if (auto resources = prob.getLinkedResourceTypes(nodes[dst].op))
      for (auto resource : *resources)
        if (prob.getLimit(resource).value_or(0) > 0 &&
            !llvm::is_contained(nodes[dst].limitedResources, resource)) {
          nodes[dst].limitedResources.push_back(resource);
          ++demand[resource];
        }
  }

  // Extract the same-iteration dependence DAG. Loop-carried dependences are
  // essential to modulo feasibility, but they point between iterations and
  // therefore do not describe which operations directly delay this
  // iteration's requested objective. Use the zero-distance subgraph to seed
  // the policy with the same inexpensive criticality features as the acyclic
  // scheduler.
  SmallVector<SmallVector<unsigned, 4>> zeroDistanceSuccessors(nodes.size());
  SmallVector<unsigned> zeroDistanceIndegree(nodes.size(), 0);
  for (const auto &constraint : dependenceConstraints) {
    if (constraint.distance != 0)
      continue;
    zeroDistanceSuccessors[constraint.source].push_back(constraint.destination);
    ++zeroDistanceIndegree[constraint.destination];
  }

  SmallVector<unsigned> zeroDistanceOrder;
  SmallVector<unsigned> ready;
  for (unsigned node = 0, e = nodes.size(); node != e; ++node)
    if (zeroDistanceIndegree[node] == 0)
      ready.push_back(node);
  for (size_t cursor = 0; cursor != ready.size(); ++cursor) {
    unsigned node = ready[cursor];
    zeroDistanceOrder.push_back(node);
    for (unsigned successor : zeroDistanceSuccessors[node])
      if (--zeroDistanceIndegree[successor] == 0)
        ready.push_back(successor);
  }

  // A feasible zero-latency cycle may make the zero-distance graph cyclic.
  // In that unusual case, leave the criticality features neutral; the
  // difference solver still handles the actual recurrence constraints.
  if (zeroDistanceOrder.size() == nodes.size()) {
    for (unsigned node : zeroDistanceOrder)
      for (unsigned successor : zeroDistanceSuccessors[node])
        nodes[successor].asap = std::max(
            nodes[successor].asap, nodes[node].asap + nodes[node].latency);

    nodes[lastNode].reachesObjective = true;
    for (unsigned node : llvm::reverse(zeroDistanceOrder)) {
      if (node == lastNode)
        continue;
      for (unsigned successor : zeroDistanceSuccessors[node]) {
        if (!nodes[successor].reachesObjective)
          continue;
        nodes[node].reachesObjective = true;
        nodes[node].height = std::max(
            nodes[node].height, nodes[node].latency + nodes[successor].height);
      }
    }

    SmallVector<unsigned> latest(nodes.size(),
                                 std::numeric_limits<unsigned>::max());
    latest[lastNode] = nodes[lastNode].asap;
    for (unsigned node : llvm::reverse(zeroDistanceOrder)) {
      if (node == lastNode || !nodes[node].reachesObjective)
        continue;
      for (unsigned successor : zeroDistanceSuccessors[node]) {
        if (!nodes[successor].reachesObjective ||
            latest[successor] == std::numeric_limits<unsigned>::max())
          continue;
        assert(latest[successor] >= nodes[node].latency &&
               "same-iteration objective must admit an ALAP schedule");
        latest[node] =
            std::min(latest[node], latest[successor] - nodes[node].latency);
      }
      assert(latest[node] != std::numeric_limits<unsigned>::max() &&
             "objective fan-in must have a latest start time");
      nodes[node].mobility = latest[node] - nodes[node].asap;
    }
  }

  unsigned maxHeight = 0, maxMobility = 0, maxLatency = 0, maxFanout = 0;
  double maxPressure = 0.0;
  SmallVector<double> pressure(nodes.size(), 0.0);
  for (auto [index, node] : llvm::enumerate(nodes)) {
    maxHeight = std::max(maxHeight, node.height);
    maxMobility = std::max(maxMobility, node.mobility);
    maxLatency = std::max(maxLatency, node.latency);
    maxFanout = std::max<unsigned>(maxFanout, node.successors.size());
    for (auto resource : node.limitedResources)
      pressure[index] = std::max(pressure[index],
                                 static_cast<double>(demand.lookup(resource)) /
                                     *prob.getLimit(resource));
    maxPressure = std::max(maxPressure, pressure[index]);
  }
  for (auto [index, node] : llvm::enumerate(nodes)) {
    node.features[Bias] = 1.0;
    node.features[ReachesObjective] = node.reachesObjective ? 1.0 : 0.0;
    node.features[CriticalPathHeight] =
        maxHeight == 0 ? 0.0 : static_cast<double>(node.height) / maxHeight;
    node.features[TightMobility] =
        node.reachesObjective ? 1.0 - static_cast<double>(node.mobility) /
                                          std::max(1u, maxMobility)
                              : 0.0;
    node.features[Fanout] =
        maxFanout == 0
            ? 0.0
            : static_cast<double>(node.successors.size()) / maxFanout;
    node.features[Latency] =
        maxLatency == 0 ? 0.0 : static_cast<double>(node.latency) / maxLatency;
    node.features[ResourcePressure] =
        maxPressure == 0.0 ? 0.0 : pressure[index] / maxPressure;
  }
  return buildClusters();
}

LogicalResult ModuloNodeRLScheduler::buildClusters() {
  DisjointSet sets(nodes.size());

  // A limited resource is a global capacity constraint. All of its users must
  // therefore remain in the same cluster. An operation using several limited
  // resources joins those resource groups transitively.
  DenseMap<Problem::ResourceType, unsigned> firstResourceUser;
  for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
    for (auto resource : nodes[node].limitedResources) {
      auto first = firstResourceUser.find(resource);
      if (first == firstResourceUser.end()) {
        firstResourceUser[resource] = node;
        continue;
      }
      sets.unite(node, first->second);
    }
  }

  // Construct the dependence graph after resource groups have been joined.
  // Its SCCs are the remaining indivisible units: splitting one would leave a
  // cycle between clusters, which cannot be repaired with a one-way offset.
  DenseMap<unsigned, unsigned> vertexForRoot;
  SmallVector<unsigned> vertexRepresentatives;
  SmallVector<unsigned> vertexOfNode(nodes.size());
  for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
    unsigned root = sets.find(node);
    auto vertex = vertexForRoot.find(root);
    if (vertex == vertexForRoot.end()) {
      unsigned index = vertexRepresentatives.size();
      vertexForRoot[root] = index;
      vertexRepresentatives.push_back(node);
      vertexOfNode[node] = index;
    } else {
      vertexOfNode[node] = vertex->second;
    }
  }

  SmallVector<SmallVector<unsigned, 4>> adjacency(vertexRepresentatives.size());
  SmallVector<SmallVector<unsigned, 4>> reverseAdjacency(
      vertexRepresentatives.size());
  for (const auto &constraint : dependenceConstraints) {
    unsigned source = vertexOfNode[constraint.source];
    unsigned destination = vertexOfNode[constraint.destination];
    if (source == destination)
      continue;
    adjacency[source].push_back(destination);
    reverseAdjacency[destination].push_back(source);
  }

  // Iterative Kosaraju traversal avoids recursion depth proportional to a
  // large scheduling graph.
  struct DFSFrame {
    unsigned vertex;
    unsigned nextSuccessor;
  };
  SmallVector<char> visited(vertexRepresentatives.size(), false);
  SmallVector<unsigned> finishOrder;
  for (unsigned start = 0, e = vertexRepresentatives.size(); start != e;
       ++start) {
    if (visited[start])
      continue;
    SmallVector<DFSFrame> stack{{start, 0}};
    visited[start] = true;
    while (!stack.empty()) {
      auto &frame = stack.back();
      if (frame.nextSuccessor != adjacency[frame.vertex].size()) {
        unsigned successor = adjacency[frame.vertex][frame.nextSuccessor++];
        if (!visited[successor]) {
          visited[successor] = true;
          stack.push_back({successor, 0});
        }
        continue;
      }
      finishOrder.push_back(frame.vertex);
      stack.pop_back();
    }
  }

  llvm::fill(visited, false);
  for (unsigned start : llvm::reverse(finishOrder)) {
    if (visited[start])
      continue;
    SmallVector<unsigned> stack{start};
    visited[start] = true;
    unsigned representative = vertexRepresentatives[start];
    while (!stack.empty()) {
      unsigned vertex = stack.pop_back_val();
      sets.unite(representative, vertexRepresentatives[vertex]);
      for (unsigned predecessor : reverseAdjacency[vertex]) {
        if (visited[predecessor])
          continue;
        visited[predecessor] = true;
        stack.push_back(predecessor);
      }
    }
  }

  // Give the final clusters and their local nodes stable insertion-order
  // indices. Local numbering keeps the difference solvers proportional to the
  // cluster size instead of the whole problem size.
  DenseMap<unsigned, unsigned> clusterForRoot;
  clusterOfNode.resize(nodes.size());
  localIndexOfNode.resize(nodes.size());
  for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
    unsigned root = sets.find(node);
    auto cluster = clusterForRoot.find(root);
    unsigned clusterIndex;
    if (cluster == clusterForRoot.end()) {
      clusterIndex = clusters.size();
      clusterForRoot[root] = clusterIndex;
      clusters.emplace_back();
    } else {
      clusterIndex = cluster->second;
    }
    clusterOfNode[node] = clusterIndex;
    localIndexOfNode[node] = clusters[clusterIndex].nodes.size();
    clusters[clusterIndex].nodes.push_back(node);
  }

  outgoingClusterConstraints.resize(clusters.size());
  SmallVector<unsigned> indegree(clusters.size(), 0);
  for (const auto &constraint : dependenceConstraints) {
    unsigned sourceCluster = clusterOfNode[constraint.source];
    unsigned destinationCluster = clusterOfNode[constraint.destination];
    if (sourceCluster == destinationCluster) {
      clusters[sourceCluster].dependenceConstraints.push_back(
          {localIndexOfNode[constraint.source],
           localIndexOfNode[constraint.destination], constraint.delay,
           constraint.distance});
      continue;
    }
    outgoingClusterConstraints[sourceCluster].push_back(constraint);
    ++indegree[destinationCluster];
  }

  // Record only the users local to each resource cluster. This also removes a
  // full-graph scan from every resource-conflict decision.
  for (auto resource : prob.getResourceTypes()) {
    SmallVector<unsigned> resourceNodes;
    std::optional<unsigned> resourceCluster;
    for (unsigned node = 0, e = nodes.size(); node != e; ++node) {
      if (!llvm::is_contained(nodes[node].limitedResources, resource))
        continue;
      unsigned cluster = clusterOfNode[node];
      assert((!resourceCluster || *resourceCluster == cluster) &&
             "resource users must have been joined into one cluster");
      resourceCluster = cluster;
      resourceNodes.push_back(localIndexOfNode[node]);
    }
    if (resourceCluster)
      clusters[*resourceCluster].resources.push_back(
          {resource, std::move(resourceNodes)});
  }

  SmallVector<unsigned> ready;
  for (unsigned cluster = 0, e = clusters.size(); cluster != e; ++cluster)
    if (indegree[cluster] == 0)
      ready.push_back(cluster);
  for (size_t cursor = 0; cursor != ready.size(); ++cursor) {
    unsigned cluster = ready[cursor];
    clusterTopologicalOrder.push_back(cluster);
    for (const auto &constraint : outgoingClusterConstraints[cluster]) {
      unsigned destination = clusterOfNode[constraint.destination];
      assert(indegree[destination] != 0 && "invalid cluster indegree");
      if (--indegree[destination] == 0)
        ready.push_back(destination);
    }
  }
  if (clusterTopologicalOrder.size() != clusters.size())
    return prob.getContainingOp()->emitError(
        "node-rl could not form an acyclic scheduling-cluster graph");

  LLVM_DEBUG({
    size_t maxClusterSize = 0;
    for (const auto &cluster : clusters)
      maxClusterSize = std::max(maxClusterSize, cluster.nodes.size());
    llvm::dbgs() << "node-rl modulo formed " << clusters.size()
                 << " scheduling clusters; largest=" << maxClusterSize << "\n";
  });
  return success();
}

bool ModuloNodeRLScheduler::solveDifferenceConstraints(
    ArrayRef<ModuloConstraint> constraints, unsigned numNodes, unsigned ii,
    SmallVectorImpl<int64_t> &startTimes) const {
  startTimes.assign(numNodes, 0);
  for (unsigned iteration = 0; iteration != numNodes; ++iteration) {
    bool changed = false;
    for (const auto &constraint : constraints) {
      int64_t required = startTimes[constraint.source] + constraint.delay -
                         constraint.distance * static_cast<int64_t>(ii);
      if (required <= startTimes[constraint.destination])
        continue;
      startTimes[constraint.destination] = required;
      changed = true;
    }
    if (!changed)
      return true;
    // A relaxation during the nth pass identifies a positive cycle, i.e. an
    // infeasible recurrence or a resource order incompatible with this II.
    if (iteration + 1 == numNodes)
      return false;
  }
  return true;
}

ModuloConstraint ModuloNodeRLScheduler::orderResourceConflict(
    const ModuloCluster &cluster, ResourceConflict conflict,
    ArrayRef<int64_t> startTimes, unsigned ii, const FeatureVector &weights,
    bool stochastic, double temperature, bool wrapAround,
    ModuloClusterEpisode &episode) {
  unsigned source = conflict.source;
  unsigned destination = conflict.destination;
  bool advanceAcrossWrap = false;
  if (conflict.sourcePhase == conflict.destinationPhase) {
    SmallVector<double> scores;
    for (unsigned node : {source, destination}) {
      unsigned globalNode = cluster.nodes[node];
      double score = 0.0;
      for (unsigned feature = 0; feature != NumFeatures; ++feature)
        score += weights[feature] * nodes[globalNode].features[feature];
      scores.push_back(score);
    }
    unsigned selected = scores[1] > scores[0] ? 1 : 0;
    if (stochastic) {
      temperature = std::max(temperature, 1.0e-6);
      double p0 = 1.0 / (1.0 + std::exp((scores[1] - scores[0]) / temperature));
      std::bernoulli_distribution distribution(p0);
      selected = distribution(rng) ? 0 : 1;
      for (unsigned feature = 0; feature != NumFeatures; ++feature) {
        double expected =
            p0 * nodes[cluster.nodes[source]].features[feature] +
            (1.0 - p0) * nodes[cluster.nodes[destination]].features[feature];
        episode.policyGradient[feature] +=
            (nodes[cluster.nodes[selected == 0 ? source : destination]]
                 .features[feature] -
             expected) /
            temperature;
      }
      ++episode.decisions;
    }
    if (wrapAround && !stochastic)
      selected = 1 - selected;
    if (selected == 1)
      std::swap(source, destination);
  } else {
    // The reservation windows overlap even though their start phases differ.
    // The ordinary repair advances across the shorter, conflicting circular
    // gap. A fallback episode may instead reverse this individual ordering
    // and advance across the modulo boundary. Sampling per conflict permits a
    // schedule to mix both orientations instead of choosing one globally.
    unsigned sourcePhase = static_cast<uint64_t>(startTimes[source]) % ii;
    unsigned destinationPhase =
        static_cast<uint64_t>(startTimes[destination]) % ii;
    unsigned forward =
        (static_cast<uint64_t>(destinationPhase) + ii - sourcePhase) % ii;
    if (forward >= conflict.resourceII)
      std::swap(source, destination);

    bool reverseOrdering = wrapAround;
    if (wrapAround && stochastic) {
      std::bernoulli_distribution distribution(0.5);
      reverseOrdering = distribution(rng);
      ++episode.decisions;
    }
    if (reverseOrdering) {
      std::swap(source, destination);
      advanceAcrossWrap = true;
    }
  }

  // Absolute start times may differ by whole IIs while their phases alias. A
  // plain `source -> destination` edge with delay=resourceII is then redundant
  // (for example starts 3 and 6 at II=3). Preserve the current iteration
  // alignment in the pseudo-edge, so it advances the destination phase by the
  // exact amount needed to clear the reservation window. Negative distances
  // are internal resource-ordering offsets, not loop-carried dependences.
  int64_t difference = startTimes[destination] - startTimes[source];
  int64_t iterationDelta = difference / static_cast<int64_t>(ii);
  if (difference < 0 && difference % static_cast<int64_t>(ii) != 0)
    --iterationDelta;
  if (advanceAcrossWrap)
    ++iterationDelta;
  return ModuloConstraint{source, destination, conflict.resourceII,
                          -iterationDelta};
}

bool ModuloNodeRLScheduler::shouldTrackReservations(
    const ModuloCluster &cluster) const {
  // Rebuilding flat reservation buckets is faster for the small-capacity
  // clusters common in HLS. A large shared pool is different: resolving one
  // overflow at a time otherwise rescans hundreds or thousands of users. Use
  // lazy incremental heaps for that high-capacity regime. A low-capacity
  // group with dozens of users has the same aggregate problem even though each
  // conflict bucket is small, so bulk-order it once pairwise conflict repair
  // would begin to dominate the work budget.
  bool hasCoupledResourceUse = llvm::any_of(cluster.nodes, [&](unsigned node) {
    return nodes[node].limitedResources.size() > 1;
  });
  return llvm::any_of(cluster.resources, [&](const ClusterResource &resource) {
    return resource.nodes.size() >= 32 ||
           (resource.nodes.size() >= 16 &&
            prob.getLimit(resource.resource).value_or(0) >= 4) ||
           (hasCoupledResourceUse && resource.nodes.size() >= 8);
  });
}

bool ModuloNodeRLScheduler::consumeResourceOrdering() {
  if (options.resourceOrderingBudget != 0 &&
      resourceOrderings >= options.resourceOrderingBudget) {
    resourceOrderingBudgetExhausted = true;
    return false;
  }
  ++resourceOrderings;
  return true;
}

std::optional<ModuloConstraint> ModuloNodeRLScheduler::chooseResourceConstraint(
    const ModuloCluster &cluster, ArrayRef<int64_t> startTimes, unsigned ii,
    const FeatureVector &weights, bool stochastic, double temperature,
    bool wrapAround, ModuloClusterEpisode &episode) {
  for (const auto &resourceUse : cluster.resources) {
    auto resource = resourceUse.resource;
    auto limit = prob.getLimit(resource).value_or(0);
    assert(limit != 0 && "clusters must contain only limited resources");
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    SmallVector<SmallVector<unsigned, 4>> reservations(ii);
    for (unsigned node : resourceUse.nodes) {
      for (unsigned offset = 0; offset != resourceII; ++offset) {
        reservations[(startTimes[node] + offset) % ii].push_back(node);
      }
    }
    for (auto &active : reservations) {
      if (active.size() <= limit)
        continue;
      llvm::sort(active, [&](unsigned lhs, unsigned rhs) {
        return std::make_tuple(startTimes[lhs] % ii, cluster.nodes[lhs]) <
               std::make_tuple(startTimes[rhs] % ii, cluster.nodes[rhs]);
      });
      unsigned source = active[0], destination = active[1];
      return orderResourceConflict(
          cluster,
          {source, destination, static_cast<unsigned>(startTimes[source] % ii),
           static_cast<unsigned>(startTimes[destination] % ii), resourceII},
          startTimes, ii, weights, stochastic, temperature, wrapAround,
          episode);
    }
  }
  return std::nullopt;
}

bool ModuloNodeRLScheduler::bindResources(ArrayRef<int64_t> startTimes,
                                          unsigned ii) {
  // `ResourceBindingsProp` names one fixed instance per operation. If a
  // resource accepts requests less frequently than the pipeline II, a single
  // operation needs to rotate through multiple instances across iterations.
  // The modulo reservation table and verifier model that capacity exactly, but
  // the current property cannot encode the rotation. Leave all bindings absent
  // in this case; a downstream implementation must synthesize the rotating
  // arbitration from the reservation schedule.
  for (auto resource : prob.getResourceTypes())
    if (prob.getLimit(resource).value_or(0) > 0 &&
        prob.getResourceInitiationInterval(resource).value_or(1) > ii) {
      prob.clearResourceBindings();
      return true;
    }

  SmallVector<SmallVector<unsigned, 2>> bindings(nodes.size());
  for (unsigned node = 0, e = nodes.size(); node != e; ++node)
    bindings[node].resize(nodes[node].limitedResources.size());

  for (auto resource : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(resource).value_or(0);
    if (limit == 0)
      continue;
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    if (resourceII > ii)
      return false;
    SmallVector<unsigned> uses;
    for (unsigned node = 0, e = nodes.size(); node != e; ++node)
      if (llvm::is_contained(nodes[node].limitedResources, resource))
        uses.push_back(node);
    llvm::sort(uses, [&](unsigned lhs, unsigned rhs) {
      return std::make_tuple(startTimes[lhs] % ii, lhs) <
             std::make_tuple(startTimes[rhs] % ii, rhs);
    });
    SmallVector<SmallVector<unsigned, 4>> lanes(limit);
    for (unsigned node : uses) {
      unsigned phase = startTimes[node] % ii;
      unsigned lane = 0;
      for (; lane != lanes.size(); ++lane)
        if (lanes[lane].empty() || phase >= lanes[lane].back() + resourceII)
          break;
      if (lane == lanes.size()) {
        // The modulo reservation table is feasible, but its circular-arc
        // coloring may need a rotating assignment that ResourceBindingsProp
        // cannot express. Keep the valid dynamic reservation schedule instead
        // of rejecting it merely because no static binding was found.
        prob.clearResourceBindings();
        return true;
      }
      lanes[lane].push_back(phase);
      auto resourceIndex = llvm::find(nodes[node].limitedResources, resource) -
                           nodes[node].limitedResources.begin();
      bindings[node][resourceIndex] = lane;
    }
    for (const auto &lane : lanes)
      if (!lane.empty() && lane.front() + ii < lane.back() + resourceII) {
        prob.clearResourceBindings();
        return true;
      }
  }

  prob.clearResourceBindings();
  for (unsigned node = 0, e = nodes.size(); node != e; ++node)
    if (!bindings[node].empty())
      prob.setResourceBindings(nodes[node].op, std::move(bindings[node]));
  return true;
}

ModuloClusterEpisode ModuloNodeRLScheduler::tryBulkResourceOrdering(
    const ModuloCluster &cluster, unsigned ii, ArrayRef<int64_t> baseStartTimes,
    const FeatureVector &weights, bool stochastic, double temperature) {
  ModuloClusterEpisode episode;
  IncrementalDifferenceSolver solver(
      cluster.nodes.size(), ii, cluster.dependenceConstraints, baseStartTimes);

  SmallVector<double> priorities(cluster.nodes.size());
  std::uniform_real_distribution<double> uniform(std::nextafter(0.0, 1.0), 1.0);
  for (unsigned localNode = 0, e = cluster.nodes.size(); localNode != e;
       ++localNode) {
    unsigned globalNode = cluster.nodes[localNode];
    double priority = 0.0;
    for (unsigned feature = 0; feature != NumFeatures; ++feature)
      priority += weights[feature] * nodes[globalNode].features[feature];
    if (stochastic) {
      // Gumbel perturbations sample a policy-ranked order without repeatedly
      // materializing a softmax over a potentially huge equal-phase set.
      double sample = uniform(rng);
      priority +=
          std::max(temperature, 1.0e-6) * (-std::log(-std::log(sample)));
    }
    priorities[localNode] = priority;
  }

  for (const auto &resourceUse : cluster.resources) {
    unsigned limit = *prob.getLimit(resourceUse.resource);
    unsigned resourceII =
        prob.getResourceInitiationInterval(resourceUse.resource).value_or(1);
    unsigned laneCount = std::min<unsigned>(limit, resourceUse.nodes.size());
    if (laneCount == 0)
      continue;

    SmallVector<unsigned> order(resourceUse.nodes.begin(),
                                resourceUse.nodes.end());
    llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
      unsigned lhsPhase = static_cast<uint64_t>(baseStartTimes[lhs]) % ii;
      unsigned rhsPhase = static_cast<uint64_t>(baseStartTimes[rhs]) % ii;
      if (lhsPhase != rhsPhase)
        return lhsPhase < rhsPhase;
      if (priorities[lhs] != priorities[rhs])
        return priorities[lhs] > priorities[rhs];
      return cluster.nodes[lhs] < cluster.nodes[rhs];
    });

    SmallVector<SmallVector<unsigned>> lanes(laneCount);
    for (auto [position, node] : llvm::enumerate(order))
      lanes[position % laneCount].push_back(node);
    for (const auto &lane : lanes) {
      if (static_cast<uint64_t>(lane.size()) * resourceII > ii)
        return episode;
      for (unsigned index = 1; index != lane.size(); ++index) {
        if (!consumeResourceOrdering())
          return episode;
        if (!solver.addConstraint(
                {lane[index - 1], lane[index], resourceII, 0}))
          return episode;
      }
      if (!consumeResourceOrdering())
        return episode;
      if (!solver.addConstraint({lane.back(), lane.front(), resourceII, 1}))
        return episode;
    }
  }

  ClusterReservationState reservations(cluster, solver.getStartTimes(), ii,
                                       prob);
  if (reservations.findConflict())
    return episode;
  episode.startTimes.assign(solver.getStartTimes().begin(),
                            solver.getStartTimes().end());
  episode.feasible = true;
  return episode;
}

uint64_t
ModuloNodeRLScheduler::countResourceExcess(const ModuloCluster &cluster,
                                           ArrayRef<int64_t> startTimes,
                                           unsigned ii) const {
  uint64_t excess = 0;
  SmallVector<unsigned> occupancy(ii);
  for (const auto &resourceUse : cluster.resources) {
    llvm::fill(occupancy, 0);
    unsigned limit = *prob.getLimit(resourceUse.resource);
    unsigned resourceII =
        prob.getResourceInitiationInterval(resourceUse.resource).value_or(1);
    for (unsigned node : resourceUse.nodes) {
      assert(startTimes[node] >= 0 &&
             "difference solution must be nonnegative");
      unsigned phase = static_cast<uint64_t>(startTimes[node]) % ii;
      for (unsigned offset = 0; offset != resourceII; ++offset)
        ++occupancy[(static_cast<uint64_t>(phase) + offset) % ii];
    }
    for (unsigned active : occupancy)
      if (active > limit)
        excess += active - limit;
  }
  return excess;
}

ModuloConstraint ModuloNodeRLScheduler::orderResourceConflictAlternative(
    ResourceConflict conflict, ArrayRef<int64_t> startTimes, unsigned ii,
    bool alternative) const {
  unsigned source = conflict.source;
  unsigned destination = conflict.destination;
  bool advanceAcrossWrap = false;
  if (conflict.sourcePhase == conflict.destinationPhase) {
    if (alternative)
      std::swap(source, destination);
  } else {
    unsigned sourcePhase = static_cast<uint64_t>(startTimes[source]) % ii;
    unsigned destinationPhase =
        static_cast<uint64_t>(startTimes[destination]) % ii;
    unsigned forward =
        (static_cast<uint64_t>(destinationPhase) + ii - sourcePhase) % ii;
    if (forward >= conflict.resourceII)
      std::swap(source, destination);
    if (alternative) {
      std::swap(source, destination);
      advanceAcrossWrap = true;
    }
  }

  int64_t difference = startTimes[destination] - startTimes[source];
  int64_t iterationDelta = difference / static_cast<int64_t>(ii);
  if (difference < 0 && difference % static_cast<int64_t>(ii) != 0)
    --iterationDelta;
  if (advanceAcrossWrap)
    ++iterationDelta;
  return {source, destination, conflict.resourceII, -iterationDelta};
}

bool ModuloNodeRLScheduler::applyMCTSAction(ModuloSearchState &state,
                                            const ModuloConstraint &action) {
  ModuloConstraintKey key = getConstraintKey(action);
  if (llvm::is_contained(state.constraintKeys, key))
    return false;
  if (!consumeResourceOrdering())
    return false;
  state.constraintKeys.push_back(key);
  return state.solver.addConstraint(action);
}

bool ModuloNodeRLScheduler::collectMCTSActions(
    const ModuloCluster &cluster, const ModuloSearchState &state, unsigned ii,
    unsigned conflictWidth, bool maskIllegal,
    SmallVectorImpl<ModuloConstraint> &actions) {
  actions.clear();
  SmallVector<RankedResourceConflict, 16> conflicts;
  llvm::SmallDenseSet<std::tuple<unsigned, unsigned, unsigned>, 16> seenPairs;
  auto startTimes = state.solver.getStartTimes();
  for (auto [resourceIndex, resourceUse] : llvm::enumerate(cluster.resources)) {
    unsigned limit = *prob.getLimit(resourceUse.resource);
    unsigned resourceII =
        prob.getResourceInitiationInterval(resourceUse.resource).value_or(1);
    SmallVector<SmallVector<unsigned, 4>> reservations(ii);
    for (unsigned node : resourceUse.nodes) {
      unsigned phase = static_cast<uint64_t>(startTimes[node]) % ii;
      for (unsigned offset = 0; offset != resourceII; ++offset)
        reservations[(static_cast<uint64_t>(phase) + offset) % ii].push_back(
            node);
    }
    for (auto [activePhase, active] : llvm::enumerate(reservations)) {
      if (active.size() <= limit)
        continue;
      llvm::sort(active, [&](unsigned lhs, unsigned rhs) {
        return std::make_tuple(startTimes[lhs] % ii, cluster.nodes[lhs]) <
               std::make_tuple(startTimes[rhs] % ii, cluster.nodes[rhs]);
      });
      unsigned source = active[0], destination = active[1];
      auto pairKey = std::make_tuple(static_cast<unsigned>(resourceIndex),
                                     std::min(source, destination),
                                     std::max(source, destination));
      if (!seenPairs.insert(pairKey).second)
        continue;
      conflicts.push_back(
          {{source, destination, static_cast<unsigned>(startTimes[source] % ii),
            static_cast<unsigned>(startTimes[destination] % ii), resourceII},
           static_cast<unsigned>(active.size() - limit),
           static_cast<unsigned>(resourceIndex),
           static_cast<unsigned>(activePhase)});
    }
  }
  if (conflicts.empty())
    return true;

  // Width one is the existing first-conflict policy. Wider searches retain
  // that baseline action and add the most overloaded alternatives.
  if (conflicts.size() > 1)
    llvm::sort(conflicts.begin() + 1, conflicts.end(),
               [](const RankedResourceConflict &lhs,
                  const RankedResourceConflict &rhs) {
                 if (lhs.overload != rhs.overload)
                   return lhs.overload > rhs.overload;
                 if (lhs.conflict.resourceII != rhs.conflict.resourceII)
                   return lhs.conflict.resourceII > rhs.conflict.resourceII;
                 return std::tie(lhs.resourceIndex, lhs.activePhase,
                                 lhs.conflict.source,
                                 lhs.conflict.destination) <
                        std::tie(rhs.resourceIndex, rhs.activePhase,
                                 rhs.conflict.source, rhs.conflict.destination);
               });
  if (conflicts.size() > conflictWidth)
    conflicts.resize(conflictWidth);

  llvm::SmallDenseSet<ModuloConstraintKey, 16> actionKeys;
  for (const auto &ranked : conflicts) {
    for (bool alternative : {false, true}) {
      ModuloConstraint action = orderResourceConflictAlternative(
          ranked.conflict, startTimes, ii, alternative);
      ModuloConstraintKey key = getConstraintKey(action);
      if (llvm::is_contained(state.constraintKeys, key) ||
          !actionKeys.insert(key).second)
        continue;
      actions.push_back(action);
    }
  }
  if (!maskIllegal)
    return false;

  SmallVector<MCTSActionCandidate, 16> candidates;
  for (const auto &action : actions) {
    ModuloSearchState child = state;
    if (!applyMCTSAction(child, action)) {
      if (resourceOrderingBudgetExhausted) {
        actions.clear();
        return false;
      }
      continue;
    }
    candidates.push_back(
        {action,
         countResourceExcess(cluster, child.solver.getStartTimes(), ii)});
  }
  llvm::stable_sort(candidates, [](const MCTSActionCandidate &lhs,
                                   const MCTSActionCandidate &rhs) {
    return lhs.resourceExcess < rhs.resourceExcess;
  });
  actions.clear();
  for (const auto &candidate : candidates)
    actions.push_back(candidate.action);
  return false;
}

void ModuloNodeRLScheduler::packTightUnitCapacityResources(
    const ModuloCluster &cluster, unsigned ii, ModuloSearchState &state,
    std::chrono::steady_clock::time_point deadline) {
  for (const auto &resourceUse : cluster.resources) {
    if (std::chrono::steady_clock::now() >= deadline ||
        resourceOrderingBudgetExhausted)
      return;
    if (prob.getLimit(resourceUse.resource).value_or(0) != 1)
      continue;
    unsigned resourceII =
        prob.getResourceInitiationInterval(resourceUse.resource).value_or(1);
    if (resourceUse.nodes.empty() ||
        static_cast<uint64_t>(resourceUse.nodes.size()) * resourceII != ii)
      continue;

    ModuloSearchState packed = state;
    SmallVector<unsigned> order(resourceUse.nodes.begin(),
                                resourceUse.nodes.end());
    auto startTimes = packed.solver.getStartTimes();
    llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
      unsigned lhsPhase = static_cast<uint64_t>(startTimes[lhs]) % ii;
      unsigned rhsPhase = static_cast<uint64_t>(startTimes[rhs]) % ii;
      return std::tie(lhsPhase, startTimes[lhs], cluster.nodes[lhs]) <
             std::tie(rhsPhase, startTimes[rhs], cluster.nodes[rhs]);
    });

    bool feasible = true;
    for (unsigned index = 0, e = order.size(); index != e; ++index) {
      if (std::chrono::steady_clock::now() >= deadline) {
        feasible = false;
        break;
      }
      ModuloConstraint ordering{order[index], order[(index + 1) % e],
                                resourceII, index + 1 == e ? 1 : 0};
      if (llvm::is_contained(packed.constraintKeys, getConstraintKey(ordering)))
        continue;
      if (!applyMCTSAction(packed, ordering)) {
        feasible = false;
        break;
      }
    }
    if (feasible)
      state = std::move(packed);
  }
}

ModuloClusterEpisode ModuloNodeRLScheduler::tryMCTSResourceOrdering(
    const ModuloCluster &cluster, unsigned ii, ArrayRef<int64_t> baseStartTimes,
    std::chrono::steady_clock::time_point deadline) {
  ModuloClusterEpisode episode;
  ModuloSearchState root(cluster, ii, baseStartTimes);
  packTightUnitCapacityResources(cluster, ii, root, deadline);
  if (std::chrono::steady_clock::now() >= deadline ||
      resourceOrderingBudgetExhausted)
    return episode;

  uint64_t rootExcess =
      countResourceExcess(cluster, root.solver.getStartTimes(), ii);
  if (rootExcess == 0) {
    episode.startTimes.assign(root.solver.getStartTimes().begin(),
                              root.solver.getStartTimes().end());
    episode.feasible = true;
    return episode;
  }

  SmallVector<MCTSTree, 3> trees;
  trees.reserve(options.mctsRescueTrees);
  for (unsigned tree = 0; tree != options.mctsRescueTrees; ++tree) {
    constexpr uint64_t goldenRatio = 0x9e3779b97f4a7c15ULL;
    trees.emplace_back(root, options.seed + goldenRatio * (tree + 1));
    trees.back().nodes.reserve(options.mctsRescueSimulations + 1);
  }
  uint64_t rolloutLimit = std::max<uint64_t>(8, cluster.nodes.size() * 8);
  rolloutLimit =
      std::min<uint64_t>(rolloutLimit, std::numeric_limits<unsigned>::max());

  auto runSimulation = [&](MCTSTree &tree) {
    unsigned nodeIndex = 0;
    SmallVector<unsigned, 16> path{nodeIndex};
    while (true) {
      if (std::chrono::steady_clock::now() >= deadline ||
          resourceOrderingBudgetExhausted)
        return false;
      auto &node = tree.nodes[nodeIndex];
      if (!node.actionsReady) {
        node.feasible = collectMCTSActions(
            cluster, node.state, ii, options.mctsTreeWidth, true, node.actions);
        node.actionsReady = true;
        node.deadEnd = !node.feasible && node.actions.empty();
      }
      if (node.feasible) {
        episode.startTimes.assign(node.state.solver.getStartTimes().begin(),
                                  node.state.solver.getStartTimes().end());
        episode.feasible = true;
        return true;
      }
      if (node.deadEnd)
        break;

      unsigned widenedActions = std::min<unsigned>(
          node.actions.size(),
          std::max(2u,
                   2u * static_cast<unsigned>(std::sqrt(node.visits + 1.0))));
      if (node.nextAction != widenedActions) {
        ModuloConstraint action = node.actions[node.nextAction++];
        ModuloSearchState childState = node.state;
        bool legal = applyMCTSAction(childState, action);
        if (resourceOrderingBudgetExhausted)
          return false;
        unsigned childIndex = tree.nodes.size();
        unsigned childDepth = node.depth + 1;
        node.children.push_back(childIndex);
        tree.nodes.emplace_back(std::move(childState), childDepth);
        tree.nodes.back().deadEnd = !legal;
        nodeIndex = childIndex;
        path.push_back(nodeIndex);
        break;
      }

      double bestScore = -std::numeric_limits<double>::infinity();
      unsigned selectedChild = node.children.front();
      double logParent = std::log(static_cast<double>(node.visits) + 1.0);
      for (unsigned childIndex : node.children) {
        const auto &child = tree.nodes[childIndex];
        double score =
            child.visits == 0
                ? std::numeric_limits<double>::infinity()
                : child.totalReward / child.visits +
                      std::sqrt(2.0) * std::sqrt(logParent / child.visits);
        if (score > bestScore) {
          bestScore = score;
          selectedChild = childIndex;
        }
      }
      nodeIndex = selectedChild;
      path.push_back(nodeIndex);
    }

    double reward = 0.0;
    if (!tree.nodes[nodeIndex].deadEnd) {
      ModuloSearchState rollout = tree.nodes[nodeIndex].state;
      bool rolloutDeadEnd = false;
      uint64_t rolloutBestExcess =
          countResourceExcess(cluster, rollout.solver.getStartTimes(), ii);
      for (unsigned step = 0; step != rolloutLimit; ++step) {
        if (std::chrono::steady_clock::now() >= deadline ||
            resourceOrderingBudgetExhausted)
          return false;
        SmallVector<ModuloConstraint, 16> actions;
        bool feasible = collectMCTSActions(
            cluster, rollout, ii, options.mctsRolloutWidth, false, actions);
        if (feasible) {
          episode.startTimes.assign(rollout.solver.getStartTimes().begin(),
                                    rollout.solver.getStartTimes().end());
          episode.feasible = true;
          return true;
        }
        // Keep rollout-mode selection on an RNG stream separate from action
        // shuffling. The current policy is fully random, but retaining this
        // draw keeps portfolio seeds stable if a greedy mode is reintroduced.
        std::bernoulli_distribution chooseGreedy(0.0);
        (void)chooseGreedy(tree.rng);
        std::shuffle(actions.begin(), actions.end(), tree.rng);
        bool advanced = false;
        for (const auto &action : actions) {
          ModuloSearchState child = rollout;
          if (applyMCTSAction(child, action)) {
            rollout = std::move(child);
            advanced = true;
            break;
          }
          if (resourceOrderingBudgetExhausted)
            return false;
        }
        if (!advanced) {
          rolloutDeadEnd = true;
          break;
        }
        rolloutBestExcess = std::min(
            rolloutBestExcess,
            countResourceExcess(cluster, rollout.solver.getStartTimes(), ii));
      }
      double progress =
          1.0 - static_cast<double>(std::min(rootExcess, rolloutBestExcess)) /
                    rootExcess;
      double denseReward = 0.95 * progress + 0.05 / (1.0 + rolloutBestExcess);
      reward = (rolloutDeadEnd ? 0.1 : 1.0) * denseReward;
    }
    for (unsigned visited : path) {
      ++tree.nodes[visited].visits;
      tree.nodes[visited].totalReward += reward;
    }
    return false;
  };

  // Interleave the independent trees. This preserves portfolio diversity
  // under one wall-clock bound instead of allowing an unlucky first tree to
  // consume the entire rescue budget.
  for (unsigned simulation = 0; simulation != options.mctsRescueSimulations;
       ++simulation)
    for (auto &tree : trees) {
      if (runSimulation(tree))
        return episode;
      if (std::chrono::steady_clock::now() >= deadline ||
          resourceOrderingBudgetExhausted)
        return episode;
    }
  return episode;
}

ModuloEpisode ModuloNodeRLScheduler::runMCTSRescue(
    unsigned ii, ArrayRef<SmallVector<int64_t>> clusterBaseStartTimes,
    std::chrono::steady_clock::time_point deadline) {
  ModuloEpisode episode;
  SmallVector<SmallVector<int64_t>> clusterStartTimes;
  clusterStartTimes.reserve(clusters.size());
  for (auto [clusterIndex, cluster] : llvm::enumerate(clusters)) {
    ModuloClusterEpisode clusterEpisode = tryMCTSResourceOrdering(
        cluster, ii, clusterBaseStartTimes[clusterIndex], deadline);
    if (!clusterEpisode.feasible)
      return episode;
    clusterStartTimes.push_back(std::move(clusterEpisode.startTimes));
  }
  combineClusterSchedules(ii, clusterStartTimes, episode);
  return episode;
}

ModuloClusterEpisode ModuloNodeRLScheduler::tryBacktrackingResourceOrdering(
    const ModuloCluster &cluster, unsigned ii, ArrayRef<int64_t> baseStartTimes,
    const FeatureVector &weights) {
  ModuloClusterEpisode episode;
  if (cluster.nodes.size() > maxBacktrackingClusterSize)
    return episode;

  struct SearchState {
    SearchState(const ModuloCluster &cluster, unsigned ii,
                ArrayRef<int64_t> baseStartTimes)
        : solver(cluster.nodes.size(), ii, cluster.dependenceConstraints,
                 baseStartTimes) {
      for (const auto &constraint : cluster.dependenceConstraints)
        constraintKeys.emplace_back(constraint.source, constraint.destination,
                                    constraint.delay, constraint.distance);
    }

    IncrementalDifferenceSolver solver;
    SmallVector<ModuloConstraintKey, 16> constraintKeys;
  };

  SmallVector<SearchState, 16> stack;
  stack.emplace_back(cluster, ii, baseStartTimes);
  uint64_t clusterSize = cluster.nodes.size();
  unsigned stateBudget = static_cast<unsigned>(std::min<uint64_t>(
      maxBacktrackingStates,
      std::max<uint64_t>(64, clusterSize * clusterSize * 32)));
  for (unsigned explored = 0; !stack.empty() && explored != stateBudget;
       ++explored) {
    SearchState state = std::move(stack.back());
    stack.pop_back();

    SmallVector<ModuloConstraint, 2> alternatives;
    for (bool wrapAround : {false, true}) {
      ModuloClusterEpisode decision;
      auto constraint =
          chooseResourceConstraint(cluster, state.solver.getStartTimes(), ii,
                                   weights, false, 1.0, wrapAround, decision);
      if (!constraint) {
        episode.startTimes.assign(state.solver.getStartTimes().begin(),
                                  state.solver.getStartTimes().end());
        episode.feasible = true;
        return episode;
      }
      ModuloConstraintKey key{constraint->source, constraint->destination,
                              constraint->delay, constraint->distance};
      if (llvm::none_of(alternatives, [&](const ModuloConstraint &other) {
            return key == ModuloConstraintKey{other.source, other.destination,
                                              other.delay, other.distance};
          }))
        alternatives.push_back(*constraint);
    }

    // A depth-first search can retain one sibling at every level. Since each
    // sibling owns a complete incremental solver, an unlucky 32-node problem
    // can otherwise keep thousands of increasingly large solver copies alive.
    // Retain a bounded frontier and prefer the nearest-conflict orientation
    // whenever only one slot remains.
    assert(stack.size() < maxBacktrackingLiveStates &&
           "backtracking frontier must remain bounded");
    unsigned availableStates = maxBacktrackingLiveStates - stack.size();
    SmallVector<SearchState, 2> children;
    for (const auto &constraint : alternatives) {
      ModuloConstraintKey key{constraint.source, constraint.destination,
                              constraint.delay, constraint.distance};
      if (llvm::is_contained(state.constraintKeys, key))
        continue;
      if (!consumeResourceOrdering())
        return episode;
      SearchState child = state;
      child.constraintKeys.push_back(key);
      if (!child.solver.addConstraint(constraint))
        continue;
      children.push_back(std::move(child));
      if (children.size() == availableStates)
        break;
    }
    // Push the opposite branch first so the preferred child is popped next.
    for (auto &child : llvm::reverse(children))
      stack.push_back(std::move(child));
  }
  return episode;
}

ModuloClusterEpisode ModuloNodeRLScheduler::runClusterEpisode(
    const ModuloCluster &cluster, unsigned ii, ArrayRef<int64_t> baseStartTimes,
    const FeatureVector &weights, bool stochastic, double temperature,
    bool wrapAround, bool backtracking) {
  if (backtracking && cluster.nodes.size() <= maxBacktrackingClusterSize)
    return tryBacktrackingResourceOrdering(cluster, ii, baseStartTimes,
                                           weights);
  wrapAround &= cluster.nodes.size() <= maxBacktrackingClusterSize;
  if (shouldTrackReservations(cluster)) {
    ModuloClusterEpisode bulk = tryBulkResourceOrdering(
        cluster, ii, baseStartTimes, weights, stochastic, temperature);
    if (bulk.feasible || resourceOrderingBudgetExhausted)
      return bulk;
  }

  ModuloClusterEpisode episode;
  IncrementalDifferenceSolver solver(
      cluster.nodes.size(), ii, cluster.dependenceConstraints, baseStartTimes);
  std::optional<ClusterReservationState> reservationState;
  if (shouldTrackReservations(cluster))
    reservationState.emplace(cluster, baseStartTimes, ii, prob);
  llvm::SmallDenseSet<std::tuple<unsigned, unsigned, unsigned, int64_t>, 16>
      constraintKeys;
  for (const auto &constraint : cluster.dependenceConstraints)
    constraintKeys.insert({constraint.source, constraint.destination,
                           constraint.delay, constraint.distance});
  uint64_t clusterSize = cluster.nodes.size();
  unsigned maxSteps = static_cast<unsigned>(
      std::min<uint64_t>(std::numeric_limits<unsigned>::max(),
                         std::max<uint64_t>(1, clusterSize * clusterSize * 4)));
  SmallVector<unsigned> changedNodes;
  for (unsigned step = 0; step != maxSteps; ++step) {
    auto startTimes = solver.getStartTimes();
    std::optional<ModuloConstraint> conflict;
    if (reservationState) {
      if (auto resourceConflict = reservationState->findConflict())
        conflict = orderResourceConflict(cluster, *resourceConflict, startTimes,
                                         ii, weights, stochastic, temperature,
                                         wrapAround, episode);
    } else {
      conflict =
          chooseResourceConstraint(cluster, startTimes, ii, weights, stochastic,
                                   temperature, wrapAround, episode);
    }
    if (!conflict) {
      episode.startTimes.assign(startTimes.begin(), startTimes.end());
      episode.feasible = true;
      return episode;
    }
    // Adding the same ordering constraint again cannot change the difference
    // solution, so the next iteration would rediscover the same conflict. In
    // unfavorable policy samples this used to repeat until maxSteps, making a
    // single failed episode disproportionately expensive on larger graphs.
    // End this policy sample and let the next episode choose another order.
    if (!constraintKeys
             .insert({conflict->source, conflict->destination, conflict->delay,
                      conflict->distance})
             .second)
      return episode;
    if (!consumeResourceOrdering())
      return episode;
    if (!solver.addConstraint(*conflict,
                              reservationState ? &changedNodes : nullptr))
      return episode;
    if (reservationState)
      reservationState->update(changedNodes, solver.getStartTimes());
  }
  return episode;
}

bool ModuloNodeRLScheduler::solveClusterDependences(
    unsigned ii,
    SmallVectorImpl<SmallVector<int64_t>> &clusterStartTimes) const {
  clusterStartTimes.clear();
  clusterStartTimes.reserve(clusters.size());
  for (const auto &cluster : clusters) {
    SmallVector<int64_t> startTimes;
    if (!solveDifferenceConstraints(cluster.dependenceConstraints,
                                    cluster.nodes.size(), ii, startTimes))
      return false;
    clusterStartTimes.push_back(std::move(startTimes));
  }
  return true;
}

void ModuloNodeRLScheduler::combineClusterSchedules(
    unsigned ii, ArrayRef<SmallVector<int64_t>> clusterStartTimes,
    ModuloEpisode &episode) const {
  assert(clusterStartTimes.size() == clusters.size() &&
         "missing cluster schedules");
  SmallVector<int64_t> offsets(clusters.size(), 0);
  for (unsigned sourceCluster : clusterTopologicalOrder) {
    for (const auto &constraint : outgoingClusterConstraints[sourceCluster]) {
      unsigned destinationCluster = clusterOfNode[constraint.destination];
      int64_t requiredOffset =
          offsets[sourceCluster] +
          clusterStartTimes[sourceCluster]
                           [localIndexOfNode[constraint.source]] +
          constraint.delay - static_cast<int64_t>(constraint.distance) * ii -
          clusterStartTimes[destinationCluster]
                           [localIndexOfNode[constraint.destination]];
      offsets[destinationCluster] =
          std::max(offsets[destinationCluster], requiredOffset);
    }
  }

  episode.startTimes.resize(nodes.size());
  for (unsigned clusterIndex = 0, e = clusters.size(); clusterIndex != e;
       ++clusterIndex) {
    const auto &cluster = clusters[clusterIndex];
    for (auto [localNode, globalNode] : llvm::enumerate(cluster.nodes))
      episode.startTimes[globalNode] =
          clusterStartTimes[clusterIndex][localNode] + offsets[clusterIndex];
  }
  episode.objectiveStart = episode.startTimes[lastNode];
  episode.feasible = true;
}

ModuloEpisode ModuloNodeRLScheduler::runEpisode(
    unsigned ii, ArrayRef<SmallVector<int64_t>> clusterBaseStartTimes,
    const FeatureVector &weights, bool stochastic, double temperature,
    bool wrapAround, bool backtracking) {
  assert(clusterBaseStartTimes.size() == clusters.size() &&
         "missing base schedule for a cluster");
  ModuloEpisode episode;
  SmallVector<SmallVector<int64_t>> clusterStartTimes;
  clusterStartTimes.reserve(clusters.size());
  for (auto [clusterIndex, cluster] : llvm::enumerate(clusters)) {
    ModuloClusterEpisode clusterEpisode = runClusterEpisode(
        cluster, ii, clusterBaseStartTimes[clusterIndex], weights, stochastic,
        temperature, wrapAround, backtracking);
    episode.decisions += clusterEpisode.decisions;
    for (unsigned feature = 0; feature != NumFeatures; ++feature)
      episode.policyGradient[feature] += clusterEpisode.policyGradient[feature];
    if (!clusterEpisode.feasible)
      return episode;
    clusterStartTimes.push_back(std::move(clusterEpisode.startTimes));
  }
  combineClusterSchedules(ii, clusterStartTimes, episode);
  return episode;
}

unsigned ModuloNodeRLScheduler::computeLowerBound() const {
  unsigned lowerBound = 1;
  for (auto resource : prob.getResourceTypes()) {
    unsigned limit = prob.getLimit(resource).value_or(0);
    if (limit == 0)
      continue;
    unsigned demand = 0;
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    for (const auto &node : nodes)
      if (llvm::is_contained(node.limitedResources, resource))
        demand += resourceII;
    lowerBound = std::max(lowerBound, (demand + limit - 1) / limit);
  }
  return lowerBound;
}

unsigned ModuloNodeRLScheduler::computeEpisodePatience() const {
  if (options.episodeNodeBudget == 0)
    return options.episodes;
  uint64_t nodeCount = std::max<size_t>(1, nodes.size());
  uint64_t scaled =
      (static_cast<uint64_t>(options.episodeNodeBudget) + nodeCount - 1) /
      nodeCount;
  return static_cast<unsigned>(
      std::min<uint64_t>(options.episodes, std::max<uint64_t>(8, scaled)));
}

LogicalResult ModuloNodeRLScheduler::schedule() {
  if (options.episodes == 0)
    return prob.getContainingOp()->emitError(
        "node-rl requires at least one training episode");
  if (!std::isfinite(options.learningRate) || options.learningRate <= 0.0 ||
      !std::isfinite(options.exploration) || options.exploration <= 0.0)
    return prob.getContainingOp()->emitError(
        "node-rl options must be finite and positive");
  if (options.mctsRescueTrees != 0 &&
      (options.mctsRescueSimulations == 0 || options.mctsTreeWidth == 0 ||
       options.mctsRolloutWidth == 0 ||
       !std::isfinite(options.mctsTimeLimitSeconds) ||
       options.mctsTimeLimitSeconds <= 0.0))
    return prob.getContainingOp()->emitError(
        "node-rl MCTS rescue options must be finite and positive");
  if (options.localSearchNodes != 0 &&
      (!std::isfinite(options.localSearchTimeLimitSeconds) ||
       options.localSearchTimeLimitSeconds <= 0.0))
    return prob.getContainingOp()->emitError(
        "node-rl local-search time limit must be finite and positive");
#ifndef SCHEDULING_OR_TOOLS
  if (options.localSearchNodes != 0)
    return prob.getContainingOp()->emitError(
        "node-rl local search requires an OR-Tools-enabled build");
#endif
  prob.clearResourceBindings();
  if (failed(initialize()))
    return failure();

  unsigned lowerBound = computeLowerBound();
  // Resource demand alone is not a lower bound for a loop-carried recurrence.
  // Find the smallest II for which the original cyclic difference constraints
  // are feasible before adding any resource ordering constraints. Any feasible
  // recurrence bound is at most the sum of operation latencies on a simple
  // cycle; exceeding that bound means there is a positive zero-distance cycle.
  unsigned recurrenceBound = 0;
  for (const auto &node : nodes)
    recurrenceBound += node.latency;
  recurrenceBound = std::max(recurrenceBound, lowerBound);
  SmallVector<SmallVector<int64_t>> recurrenceStartTimes;
  while (!solveClusterDependences(lowerBound, recurrenceStartTimes)) {
    if (lowerBound == recurrenceBound)
      return prob.getContainingOp()->emitError(
          "node-rl found an infeasible zero-distance dependence cycle");
    ++lowerBound;
  }

  // Serializing all uses of any individual resource is a finite upper bound.
  // A resource-demand bound alone is insufficient when an operation's result
  // is consumed in a later stage of the same recurrence. For example, a
  // latency-three recurrence followed by a second use of the same resource
  // needs II=4 even if that resource has only two requests. The sum of all
  // node latencies is a conservative finite upper bound for a serial schedule.
  unsigned upperBound = std::max(lowerBound, recurrenceBound);
  for (auto resource : prob.getResourceTypes()) {
    if (prob.getLimit(resource).value_or(0) == 0)
      continue;
    unsigned demand = 0;
    unsigned resourceII =
        prob.getResourceInitiationInterval(resource).value_or(1);
    for (const auto &node : nodes)
      if (llvm::is_contained(node.limitedResources, resource))
        demand += resourceII;
    upperBound = std::max(upperBound, demand);
  }

  FeatureVector weights = {0.0, 3.0, 2.0, 1.0, 0.5, 0.5, 0.5, 0.25};
  unsigned episodePatience = computeEpisodePatience();
  bool hasSmallResourceCluster =
      llvm::any_of(clusters, [](const auto &cluster) {
        return !cluster.resources.empty() &&
               cluster.nodes.size() <= maxBacktrackingClusterSize;
      });
  auto mctsDeadline = std::chrono::steady_clock::time_point::min();
  if (options.mctsRescueTrees != 0)
    mctsDeadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(options.mctsTimeLimitSeconds));
  for (unsigned ii = lowerBound; ii <= upperBound; ++ii) {
    SmallVector<SmallVector<int64_t>> baseStartTimes;
    if (!solveClusterDependences(ii, baseStartTimes))
      continue;
    ModuloEpisode best =
        runEpisode(ii, baseStartTimes, weights, false, 1.0, false, false);
    if (hasSmallResourceCluster && !best.feasible &&
        !resourceOrderingBudgetExhausted) {
      ModuloEpisode wrapped =
          runEpisode(ii, baseStartTimes, weights, false, 1.0, true, false);
      if (wrapped.feasible)
        best = std::move(wrapped);
    }
    if (hasSmallResourceCluster && !best.feasible &&
        !resourceOrderingBudgetExhausted) {
      ModuloEpisode backtracked =
          runEpisode(ii, baseStartTimes, weights, false, 1.0, false, true);
      if (backtracked.feasible)
        best = std::move(backtracked);
    }
    if (options.mctsRescueTrees != 0 && !best.feasible &&
        !resourceOrderingBudgetExhausted &&
        std::chrono::steady_clock::now() < mctsDeadline) {
      ModuloEpisode rescued = runMCTSRescue(ii, baseStartTimes, mctsDeadline);
      if (rescued.feasible) {
        LLVM_DEBUG(llvm::dbgs() << "node-rl MCTS rescued II=" << ii << "\n");
        best = std::move(rescued);
      }
    }
    if (resourceOrderingBudgetExhausted && !best.feasible)
      return prob.getContainingOp()->emitError()
             << "node-rl exhausted its resource-ordering budget of "
             << options.resourceOrderingBudget
             << " before finding a feasible schedule";
    double averageReward = best.feasible ? -best.objectiveStart : -1.0e6;
    unsigned episodesSinceImprovement = 0;
    for (unsigned iteration = 0; iteration != options.episodes; ++iteration) {
      double progress =
          episodePatience == 1
              ? 1.0
              : static_cast<double>(std::min(iteration, episodePatience - 1)) /
                    (episodePatience - 1);
      double temperature =
          std::max(0.05, options.exploration * std::pow(0.1, progress));
      ModuloEpisode sample =
          runEpisode(ii, baseStartTimes, weights, true, temperature,
                     hasSmallResourceCluster && !best.feasible, false);
      if (resourceOrderingBudgetExhausted) {
        if (!best.feasible)
          return prob.getContainingOp()->emitError()
                 << "node-rl exhausted its resource-ordering budget of "
                 << options.resourceOrderingBudget
                 << " before finding a feasible schedule";
        break;
      }
      double reward = sample.feasible ? -sample.objectiveStart : -1.0e6;
      bool improved =
          sample.feasible &&
          (!best.feasible || sample.objectiveStart < best.objectiveStart);
      if (improved)
        best = sample;
      if (sample.decisions != 0) {
        double step =
            options.learningRate * (reward - averageReward) / sample.decisions;
        for (unsigned feature = 0; feature != NumFeatures; ++feature)
          weights[feature] = std::clamp(
              weights[feature] + step * sample.policyGradient[feature], -12.0,
              12.0);
      }
      averageReward = 0.9 * averageReward + 0.1 * reward;
      episodesSinceImprovement = improved ? 0 : episodesSinceImprovement + 1;
      if (episodesSinceImprovement >= episodePatience) {
        LLVM_DEBUG(llvm::dbgs()
                   << "node-rl modulo stopped II=" << ii << " after "
                   << iteration + 1 << " stagnant episodes\n");
        break;
      }
    }
    if (!best.feasible)
      continue;
    for (unsigned node = 0, e = nodes.size(); node != e; ++node)
      prob.setStartTime(nodes[node].op, best.startTimes[node]);
    prob.setInitiationInterval(ii);
#ifdef SCHEDULING_OR_TOOLS
    if (options.localSearchNodes != 0) {
      CPSATModuloLocalSearchOptions localSearchOptions;
      localSearchOptions.neighborhoodSize = options.localSearchNodes;
      localSearchOptions.timeLimitSeconds = options.localSearchTimeLimitSeconds;
      localSearchOptions.seed = options.seed;
      CPSATModuloLocalSearchResult localSearchResult;
      if (failed(improveModuloScheduleCPSAT(prob, lastOp, localSearchOptions,
                                            &localSearchResult)))
        return failure();
      for (unsigned node = 0, e = nodes.size(); node != e; ++node)
        best.startTimes[node] = *prob.getStartTime(nodes[node].op);
      best.objectiveStart = best.startTimes[lastNode];
      LLVM_DEBUG(llvm::dbgs()
                 << "node-rl CP-SAT neighborhood size="
                 << localSearchResult.neighborhoodSize
                 << " improved t=" << localSearchResult.initialObjective
                 << " -> " << localSearchResult.finalObjective << "\n");
    }
#endif
    if (!bindResources(best.startTimes, ii))
      return prob.getContainingOp()->emitError(
          "node-rl could not construct resource bindings for its schedule");
    if (failed(prob.verify()))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "node-rl modulo selected II=" << ii
                            << " objective t=" << best.objectiveStart << "\n");
    return success();
  }
  return prob.getContainingOp()->emitError(
      "node-rl could not find a feasible modulo schedule");
}

} // namespace

LogicalResult
scheduling::scheduleNodeRL(SharedOperatorsProblem &prob, Operation *lastOp,
                           const NodeRLSchedulerOptions &options) {
  NodeRLScheduler scheduler(prob, lastOp, options);
  return scheduler.schedule();
}

LogicalResult
scheduling::scheduleNodeRL(ModuloProblem &prob, Operation *lastOp,
                           const NodeRLSchedulerOptions &options) {
  ModuloNodeRLScheduler scheduler(prob, lastOp, options);
  return scheduler.schedule();
}

FailureOr<SmallVector<ResourceParetoPoint>>
scheduling::exploreNodeRLPareto(SharedOperatorsProblem &prob, Operation *lastOp,
                                ArrayRef<ResourceAllocation> allocations,
                                ResourceCostFunction costFunction,
                                const NodeRLSchedulerOptions &options) {
  auto schedule = [&] { return scheduleNodeRL(prob, lastOp, options); };
  return detail::exploreResourcePareto(prob, lastOp, allocations, costFunction,
                                       schedule, "node-rl");
}

FailureOr<SmallVector<ResourceParetoPoint>>
scheduling::exploreNodeRLPareto(SharedOperatorsProblem &prob, Operation *lastOp,
                                ArrayRef<ResourceAllocation> allocations,
                                const NodeRLSchedulerOptions &options) {
  auto defaultCost = [&](const ResourceAllocation &allocation) {
    uint64_t resourceCost = 0;
    for (const auto &entry : allocation.limits)
      resourceCost += static_cast<uint64_t>(entry.limit) *
                      prob.getResourceCost(entry.resource).value_or(1);
    return resourceCost;
  };
  return exploreNodeRLPareto(prob, lastOp, allocations, defaultCost, options);
}

FailureOr<SmallVector<ResourceParetoPoint>>
scheduling::exploreNodeRLPareto(ModuloProblem &prob, Operation *lastOp,
                                ArrayRef<ResourceAllocation> allocations,
                                ResourceCostFunction costFunction,
                                const NodeRLSchedulerOptions &options) {
  auto schedule = [&] { return scheduleNodeRL(prob, lastOp, options); };
  return detail::exploreResourcePareto(prob, lastOp, allocations, costFunction,
                                       schedule, "node-rl");
}

FailureOr<SmallVector<ResourceParetoPoint>>
scheduling::exploreNodeRLPareto(ModuloProblem &prob, Operation *lastOp,
                                ArrayRef<ResourceAllocation> allocations,
                                const NodeRLSchedulerOptions &options) {
  auto defaultCost = [&](const ResourceAllocation &allocation) {
    uint64_t resourceCost = 0;
    for (const auto &entry : allocation.limits)
      resourceCost += static_cast<uint64_t>(entry.limit) *
                      prob.getResourceCost(entry.resource).value_or(1);
    return resourceCost;
  };
  return exploreNodeRLPareto(prob, lastOp, allocations, defaultCost, options);
}
