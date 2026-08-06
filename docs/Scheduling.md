# Static scheduling infrastructure

Scheduling is a common concern in hardware design, for example in high-level
synthesis flows targeting an FSM+Datapath execution model ("static HLS"). This
document gives an overview of, and provides rationale for, the infrastructure in
the `circt::scheduling` namespace. At its core, it defines an **extensible
problem model** that acts as an interface between **clients** (i.e. passes that
have a need to schedule a graph-like IR) and reusable **algorithm**
implementations.

This infrastructure aims to provide:
- a library of ready-to-use problem definitions and schedulers for clients to
  hook into.
- an API to make algorithm implementations comparable and reusable.
- a mechanism to extend problem definitions to model additional concerns and
  constraints.

## Getting started

Let's walk through a simple example. Assume we want to *schedule* the
computation in the entry block of a function such as `@foo(...)` in the listing
below. This means we want to assign integer *start times* to each of the
*operations* in this untimed IR.

```mlir
func @foo(%a1 : i32, %a2 : i32, %a3 : i32, %a4 : i32) -> i32 {
  %0 = arith.addi %a1, %a2 : i32
  %1 = arith.addi %0, %a3 : i32
  %2:3 = "more.results"(%0, %1) : (i32, i32) -> (i32, i32, i32)
  %3 = arith.addi %a4, %2#1 : i32
  %4 = arith.addi %2#0, %2#2 : i32
  %5 = arith.addi %3, %3 : i32
  %6 = "more.operands"(%3, %4, %5) : (i32, i32, i32) -> i32
  return %6 : i32
}
```

Our only constraint is that an operation can start *after* its operands have
been computed. The operations in our source IR are unaware of time, so we need
to associate them with a suitable *operator type*. Operator types are an
abstraction of the target architecture onto which we want to schedule the source
IR. Here, the only *property* we need to model is their *latency*. Let's assume
that additions take 1 time step, the operations in the dummy `more.` dialect
take 3 time steps. As the return operation just passes control back to the
caller, we assume a latency of 0 time steps for it.

### Boilerplate

The scheduling infrastructure currently has three toplevel header files.

```c++
//...
#include "circt/Scheduling/Problems.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Scheduling/Utilities.h"
//...
using namespace circt::scheduling;
```

### Constructing a problem instance

Our stated goal requires solving an acyclic scheduling problem without resource
constraints, represented by the `Problem` class in the scheduling
infrastructure. We need to construct an *instance* of the problem, which serves
as a container for the problem *components* as well as their properties. The
MLIR operation passed as an argument to the `get(...)` method is used to emit
diagnostics.

```c++
auto prob = Problem::get(func);
```

Then, we set up the operator types with the latencies as discussed in the
introduction. Operator types are identified by string handles.

```c++
auto retOpr = prob.getOrInsertOperatorType("return");
prob.setLatency(retOpr, 0);
auto addOpr = prob.getOrInsertOperatorType("add");
prob.setLatency(addOpr, 1);
auto mcOpr = prob.getOrInsertOperatorType("multicycle");
prob.setLatency(mcOpr, 3);
```

Next, we register all operations that we want to consider in the problem
instance, and link them to one of the operator types.

```c++
auto &block = func.getBlocks().front();
for (auto &op : block) {
  prob.insertOperation(&op);
  if (isa<func::ReturnOp>(op))
    prob.setLinkedOperatorType(&op, retOpr);
  else if (isa<arith::AddIOp>(op))
    prob.setLinkedOperatorType(&op, addOpr);
  else
    prob.setLinkedOperatorType(&op, mcOpr);
}
```

Note that we do not have to tell the instance about the *dependences* between
the operations in this simple example because the problem model automatically
includes the SSA def-use-edges maintained by MLIR. However, we often have to
consider additional dependences that are not represented by value flow, such as
memory dependences. For these situations, so-called [auxiliary](#components)
dependences between operations are inserted explicitly into the problem:
`prob.insertDependence(srcOp, destOp)`.

### Scheduling

Before we attempt to schedule, we invoke the `check()` method, which ensures
that the constructed instance is complete and valid. For example, the check
would capture if we had forgot to set an operator type's latency. We dump the
instance to visualize the dependence graph.

```c++
auto checkRes = prob.check();
assert(succeeded(checkRes));
dumpAsDOT(prob, "sched-problem.dot");
```

![Dump of example instance](https://circt.llvm.org/includes/img/sched-instance.svg)

We use a simple list scheduler, available via the `Algorithms.h` header, to
compute a solution for the instance.

```c++
auto schedRes = scheduleASAP(prob);
assert(succeeded(schedRes));
```

### Working with the solution

The solution is now stored in the instance, and we invoke the problem's
`verify()` method to ensure that the computed start times adhere to the
precedence constraint we stated earlier, i.e. operations start after their
operands have computed their results. We can also convince ourselves of that by
dumping the instance and inspecting the solution.

```c++
auto verifRes = prob.verify();
assert(succeeded(verifRes));
dumpAsDOT(prob, "sched-solution.dot");
```

![Dump of example instance, including solution](https://circt.llvm.org/includes/img/sched-solution.svg)

To inspect the solution programmatically, we can query the instance in the
following way. Note that by convention, all getters in the problem classes
return `Optional<T>` values, but as we have already verified that the start
times for registered operations are set, we can directly dereference the values.

```c++
for (auto &op : prob.getOperations())
  llvm::dbgs() << *prob.getStartTime(&op) << "\n";
```

And that's it! For a more practical example, have a look at the
[`AffineToPipeline`](https://github.com/llvm/circt/blob/main/lib/Conversion/AffineToPipeline/AffineToPipeline.cpp)
pass.

## Extensible problem model

### Theory and terminology

Scheduling problems come in many flavors and variants in the context of hardware
design. In order to make the scheduling infrastructure as modular and flexible
as CIRCT itself, it is build on the following idea of an *extensible problem
model*:

An *instance* is comprised of *components* called *operations*, *dependences*
and *operator types*. Operations and dependences form a graph structure and
correspond to the source IR to be scheduled. Operator types encode the
characteristics of the target IR. The components as well as the instance can be
annotated with *properties*. Properties are either *input* or *solution*
properties, based on whether they are supplied by the client, or computed by the
algorithm. The values of these properties are subject to the *input constraints*
and *solution constraints*, which are a first-class concern in the model and are
intended to be strictly enforced before respectively after scheduling.

Concrete problem definitions derived from this model share the same
representation of the components, but differ in their sets of properties (and
potentially distinction of input and solution properties) and input and solution
constraints. Hence, we tie together properties and constraints to model a
specific scheduling problem. Extending one (or more!) parent problems means
inheriting or adding properties, and redefining the constraints (as these don't
always compose automatically).

A key benefit of this approach is that these problem definitions provide a
reliable contract between the clients and algorithms, making it clear which
information needs to be provided, and what kind of solution is to be expected.
Clients can therefore choose a problem definition that fits their needs, and
algorithms can *opt-in* to accepting a specific subset of problems, which they
can solve efficiently. Extensibility is ensured because new problem definitions
can be added to the infrastructure (or inside a specific lowering flow, or even
out-of-tree) without adapting any existing users.

### Implementation

See
[Problems.h](https://github.com/llvm/circt/blob/main/include/circt/Scheduling/Problems.h) /
[Problems.cpp](https://github.com/llvm/circt/blob/main/lib/Scheduling/Problems.cpp).

#### Problem definitions

The `Problem` class is currently the base of the problem hierarchy. Several
extended problems are [currently defined](#available-problem-definitions) via
virtual multiple inheritance. Upon construction, a `containingOp` is passed to
instances. This MLIR operation is currently only used to emit diagnostics, and
has no semantic meaning beyond that.

#### Components

The infrastructure uses the following representation of the problem components.

Operations are just `mlir::Operation *`s.

We distinguish two kinds of dependences, *def-use* and *auxiliary*. Def-use
dependences are part of the SSA graph maintained by MLIR, and can distinguish
specific result and operand numbers. As we expect any relevant graph-like input
IR to use this MLIR facility, instances automatically consider these edges
between registered operations. Auxiliary dependences, in contrast, only specify
a source and destination operation, and have to be explicitly added to the
instance by the client, e.g. for control or memory dependences. The
`detail::Dependence` class abstracts the differences between both kinds, in
order to offer a uniform API to iterate over dependences and query their
properties.

Lastly, operator types are identified by `mlir::StringAttr`s, in order to give
clients maximum flexibility in modeling their operator library. This may change
in the future, when a CIRCT-wide concept to model physical properties of
hardware emerges.

#### Properties

Properties can involve arbitrary data types, as long as these can be stored in
maps. Problem classes offer public getter and setter methods to access a given
components properties. Getters return optional values, in order to indicate if a
property is unset. For example, the signature of the method the queries the
computed start time is `Optional<unsigned> getStartTime(Operation *op)`.

#### Constraints

Clients call the virtual `Problem::check()` method to test any input
constraints, and `Problem::verify()` to test the solution constraints. Problem
classes are expected to override them as needed. There are no further
restrictions of how these methods are implemented, but it is recommended to
introduce helper methods that test a specific aspect and can be reused in
extended problems. In addition, it makes sense to check/verify the properties in
an order that avoids redundant tests for the presence of a particular property
as well as redundant iteration over the problem components.

## Available problem definitions

*See the linked Doxygen docs for more details.*

- [Problem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1Problem.html):
  A basic, acyclic problem at the root of the problem hierarchy. Operations are
  linked to operator types, which have integer latencies. The solution comprises
  integer start times adhering to the precedence constraints implied by the
  dependences.
- [CyclicProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1CyclicProblem.html):
  Cyclic extension of `Problem`. Its solution solution can be used to construct
  a pipelined datapath with a fixed, integer initiation interval, in which the
  execution of multiple iterations/samples/etc. may overlap. Operator types are
  assumed to be fully pipelined.
- [SharedOperatorsProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1SharedOperatorsProblem.html):
  A resource-constrained scheduling problem that corresponds to multiplexing
  multiple operations onto a pre-allocated number of operator instances. A
  resource type has a `limit` and may have an `ii` property, which is the
  number of cycles for which each issued request reserves an instance. If `ii`
  is omitted, it is one and the resource is fully pipelined. The optional
  `cost` property denotes the implementation cost of one instance and can be
  used by an outer resource-allocation exploration.
- [ModuloProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1ModuloProblem.html):
  Models an HLS classic: pipeline scheduling with loop-carried dependences and
  limited resources. Its periodic reservation check includes each resource's
  multi-cycle `ii`, and an operation may consume more than one resource type.
- [ChainingProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1ChainingProblem.html):
  Extends `Problem` to consider the accumulation of physical propagation delays
  on combinational paths along SSA dependences.
- [ChainingCyclicProblem](https://circt.llvm.org/doxygen/classcirct_1_1scheduling_1_1ChainingCyclicProblem.html):
  Extends `ChainingProblem` and `CyclicProblem` to consider the accumulation
  of physical propagation delays on combinational paths along SSA dependences
  on a cyclic scheduling problem. Note that the problem does not model
  propagation delays along inter-iteration dependences. These are commonly
  represented as auxiliary dependences, which are already excluded in the
  parent ChainingProblem. In addition, the ChainingCyclicProblem explicitly
  prohibits the use of def-use dependences with a non-zero distance.

NB: The classes listed above each model a *trait*-like aspect of scheduling.
These can be used as-is, but are also intended for mixing and matching, even
though we currently do not provide definitions for all possible combinations in
order not to pollute the infrastructure. For example, the `ChainingProblem` may
be of limited use standalone, but can serve as a parent class for a future
chaining-enabled modulo scheduling problem.

## Available schedulers

- ASAP list scheduler
  ([`ASAPScheduler.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/ASAPScheduler.cpp)):
  Solves the basic `Problem` with a worklist algorithm. This is mostly a
  problem-API demo from the viewpoint of an algorithm implementation.
- Linear programming-based schedulers
  ([`SimplexSchedulers.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/SimplexSchedulers.cpp)):
  Solves `Problem`, `CyclicProblem` and `ChainingProblem` optimally, and
  `SharedOperatorsProblem` / `ModuloProblem` with simple (not state-of-the-art!)
  heuristics. This family of schedulers shares a tailored implementation of the
  simplex algorithm, as proposed by de Dinechin. See the sources for more
  details and literature references.
- Node-level reinforcement-learning scheduler
  ([`NodeRLScheduler.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/NodeRLScheduler.cpp)):
  Solves acyclic `SharedOperatorsProblem` instances with a resource-aware list
  scheduler and `ModuloProblem` instances with a cyclic difference-constraint
  scheduler. For modulo scheduling it searches increasing IIs, models
  loop-carried edges with their dependence distance, and resolves conflicts in
  a modulo reservation table. At each resource-ordering decision, a lightweight
  linear policy chooses between contending nodes; REINFORCE updates the policy
  from the final objective latency. This is episodic Monte Carlo policy-gradient
  learning. Its default path has no value network, temporal-difference
  bootstrap, replay buffer, or neural inference. An individual pairwise choice
  resembles a contextual-bandit action, but all choices in the schedule receive
  the same delayed terminal reward, so they are not trained as independent
  bandit pulls. An exponential moving-average reward is used as the REINFORCE
  baseline. An optional fixed-II MCTS rescue searches the same exact
  difference-constraint states; it is disabled by default and does not train a
  network.

  The constraint solver, rather than the policy, computes start times and
  enforces dependences. The policy only selects a disjunctive ordering when two
  periodic reservations conflict. Each II's dependence-only solution is
  computed once. Adding a resource-ordering edge then propagates only increased
  start times through a worklist; a second relaxation of the newly added edge
  detects the positive cycle that makes that ordering infeasible. This avoids
  resolving every difference constraint from zero after every action. Internal
  resource-ordering edges have signed iteration offsets. This matters when two
  absolute starts differ by one or more whole IIs but still alias the same
  modulo phase: a zero-distance edge would be weaker than the existing data
  dependence and could never move either reservation.

  Before search, all users of a limited resource are joined, followed by SCC
  closure of the dependence quotient. These maximal resource/SCC clusters are
  solved independently. Inter-cluster constraints form a DAG and are satisfied
  by shifting whole cluster schedules, which preserves every modulo phase. A
  resource with at least 32 users, a pool of at least four instances with at
  least 16 users, or a cluster coupling several resources with at least eight
  users is first ordered in virtual lanes. The lanes are internal precedence
  restrictions, not additional hardware. If that safe restriction is
  infeasible the scheduler falls back to incremental pairwise conflict repair.
  This avoids quadratic overflow repair on large pools while retaining flexible
  ordering on small low-capacity problems.

  The best feasible episode is retained, training is reproducible with a fixed
  seed, and nodes may use multiple limited resources. If an episode proposes an
  ordering constraint it already contains, the episode terminates because it
  cannot change the current solution. The configured `episodes` is a maximum.
  Modulo training also stops after a graph-size-dependent interval without an
  improvement. `episode-node-budget` controls this deterministic stagnation
  window: its value is divided by the number of nodes, with a minimum of eight
  episodes; zero disables adaptive stopping. The default 65,536 node-episode
  budget leaves small graphs unchanged while bounding unproductive exploration
  on large graphs. `resource-ordering-budget` independently bounds the number
  of precedence decisions across all candidate IIs. Through `ssp-schedule`,
  select `scheduler=node-rl` and configure `episodes`,
  `episode-node-budget`, `resource-ordering-budget`, `seed`, `learning-rate`,
  `exploration`, and the optional `mcts-*` rescue budgets in the scheduler
  options.

  A deterministic backtracking repair is also available for clusters of at
  most 32 nodes. Each branch owns an incremental solver snapshot, so the
  explored-state budget alone is not a memory bound. The live DFS frontier is
  separately capped at 64 snapshots while the 16,384-state work budget remains
  available. On the 33-operation 2MM proxy this changed one pathological
  resource allocation from about 1.51 GB and 1.84 seconds to about 27 MB and
  0.03 seconds without changing its returned Pareto frontier.

  In an OR-Tools build, `local-search-nodes` optionally re-optimizes a bounded
  neighborhood after NodeRL finds its smallest feasible II. The neighborhood
  combines tight dependence fan-in of the sink with operations competing for
  the same resources. Nodes outside it remain fixed through boundary
  release/deadline constraints, and their periodic reservations are subtracted
  from local capacity. CP-SAT receives the incumbent as a hint, minimizes the
  sink at the fixed II, and commits only a globally verified improvement.
  `local-search-time-limit` bounds this step. Local search defaults to zero
  nodes because it requires OR-Tools and can add cost without helping already
  regular large schedules; it does not try a lower II.

  The C++ `exploreNodeRLPareto` and `exploreCPSATPareto` APIs evaluate complete
  resource allocations and return the non-dominated
  latency/implementation-cost frontier; modulo points also carry their II,
  which participates in dominance. Every returned
  `ResourceParetoPoint` owns a schedule certificate in problem insertion order:
  operation start times, representable static bindings, and explicit periodic
  reservations for rotating assignments. `applyResourceParetoPoint` validates
  the certificate against the problem and allocation, restores the selected
  schedule, and runs the normal problem verifier without invoking a scheduler
  again. Thus every frontier point remains materializable, including points
  found by a stochastic policy. Missing acyclic CP-SAT bindings are
  interval-colored deterministically. Missing modulo bindings use a greedy then
  bounded circular coloring; when no inexpensive static coloring is found, the
  certificate retains explicit rotating reservations instead. NodeRL emits a
  `bindings` property for each limited operation when a static physical
  assignment can be represented. The APIs' cost-function overloads can
  evaluate non-linear or post-synthesis estimates.

  The current `bindings` property is a static operation-to-instance mapping.
  A resource hold/II longer than the pipeline II is nevertheless scheduled
  correctly through the modulo reservation table: the operation rotates across
  physical instances in successive iterations. Such a schedule deliberately
  has no static `bindings` property. Affine-to-LoopSchedule instead preserves
  the periodic request as `circt.rotating_resource_reservations` metadata on
  the cloned operation. The same path materializes assignments for start-only
  schedulers such as CP-SAT: selector period one denotes a static allocation,
  while a longer selector rotates across iterations. Each entry records
  `resource`, launch `phase`, pipeline
  `period`, reservation `hold`, available `instances`, plus a verified
  `selector` table and `selector_period`. The coloring searches for the
  shortest feasible selector period and `instances` is the number of colors
  actually used rather than the configured upper bound. Metadata is emitted
  only when the periodic coloring succeeds. This is sufficient for a
  downstream periodic arbiter. LoopSchedule-to-Calyx implements a constrained
  case for rotating integer multiplies. It creates the selected multiplier
  instances, initializes a phase register for every pipeline invocation,
  dispatches through selector guards, and advances the phase on completion.
  Multiple operations are accepted when their selector instance sets are
  disjoint.

  This lowering boundary is covered by regression tests. A recurrence with
  `limit<2>, ii<4>` and pipeline II 3 emits `selector<[0, 1]>` with selector
  period 2 and lowers to two guarded `calyx.std_mult_pipe` cells. Two operations
  with selector sets `[0,1]` and `[2,3]` lower to a four-cell shared pool. A
  CP-SAT schedule with multiplier hold three, pipeline II three, and limit two
  derives period-one selectors `[0]` and `[1]` and lowers to exactly two
  multiplier cells. That representative also passes through `calyx-native`,
  `lower-calyx-to-hw`, `lower-seq-to-sv`, and Verilog export. This is a
  structural realization result, not a cycle-level functional claim;
  overlapping selector sets remain rejected until completion ownership is
  carried with each request. Static `hold > 1` bindings and two operations
  sharing one static instance are also rejected: the current steady-state
  `calyx.par` does not preserve their distinct stage start cycles. A single
  static owner is supported. These constraints separate a scheduling/resource
  result from a general RTL implementation claim.
  [`RotatingReservationLowering.md`](RotatingReservationLowering.md) records
  the selector-table contract and the required cycle-accurate lowering work.
  A direct Calyx-to-Verilog Verilator experiment evaluates a five-iteration
  recurrence to the expected value 243 for both a static control case and the
  two-instance rotating case. The same Calyx programs currently stall after
  `calyx-native` and `lower-calyx-to-hw`, including the non-rotating control:
  the generated FSM continuously writes a `calyx.register`, but Calyx-to-HW
  suppresses writes while that register's delayed `done` is high. This is a
  downstream native-lowering limitation rather than evidence of a rotating
  selector failure; RTL obtained through that path is treated as structural
  until the register protocol is aligned.

  `utils/run_modulo_node_rl_bench.py` generates reproducible microbenchmarks
  with loop-carried chains, multiple resources, and resource holding times.
  The `gemm`, `2mm`, `jacobi-2d`, and `covariance` modes are scheduling proxies
  for PolyBench inner loops. `--chains` models independent output tiles and
  `--unrolls` controls the scalar work retained in one modulo iteration. The
  proxy deliberately excludes affine-index expansion and memory banking. Real
  Affine loop structure and lowering are covered separately by
  `test/Conversion/AffineToLoopSchedule/polybench-inner.mlir`.

  The CSV contains scheduler status, CP-SAT's resource/dependence lower bound,
  II, sink latency, configured resource limits/cost, static-binding count,
  seed, and wall time. CP-SAT starts at that bound rather than enumerating from
  II=1. The script also emits multi-seed summaries and per-seed Pareto
  frontiers. A representative small comparison can be reproduced for each
  kernel with:

  ```sh
  python3 utils/run_modulo_node_rl_bench.py \
    --circt-opt build-fc092/bin/circt-opt --kernel KERNEL \
    --chains 4 --unrolls 2 --episodes 32 256 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --mul-limit 2 --mul-ii 3 --div-limit 1 --div-ii 2 \
    --cpsat --timeout 15
  ```

  The following snapshot uses the 32-episode rows. `L min/median/max` is over
  ten NodeRL seeds. CP-SAT runs once and proves the lexicographic minimum II and
  then sink latency. The LNS column uses 16 nodes and 0.2 seconds, except
  Jacobi-2D, which uses 32 nodes and one second. Times are mean wall times.

  | held-resource proxy | CP-SAT optimum | NodeRL | NodeRL + local CP-SAT |
  | --- | --- | --- | --- |
  | GEMM / 17 nodes | II 16, L 17; 0.128 s | II 16, L 18/19/19; <0.06 s | II 16, L 17/17/18; 0.075 s |
  | 2MM / 33 nodes | II 32, L 33; 0.377 s | II 32, L 33/33/34; <0.04 s | II 32, L 33/33/34; 0.068 s |
  | Jacobi-2D / 33 nodes | II 32, L 33; 0.426 s | II 32, L 38/40.5/44; <0.05 s | II 32, L 33/33.5/36; 1.026 s |
  | Covariance / 25 nodes | II 32, L 27; 0.283 s | II 32, L 32/32/32; <0.06 s | II 32, L 27/27/27; 0.078 s |

  Raising the episode maximum to 256 changes median latency to 18, 33, 39.5,
  and 32 respectively, so additional Monte Carlo samples do not reliably close
  the nonlocal latency gap. A full 33-node, two-second Jacobi neighborhood
  reaches L=33 in all ten seeds in 2.049 seconds on average. That is effectively
  a fixed-II exact solve and is slower than full CP-SAT on this small graph;
  bounded neighborhoods are intended for graphs where the global model is no
  longer cheap. LNS never changes II, so a primary-II miss must be repaired by
  better structural ordering or by running the exact scheduler.

  The existing fully-pipelined, single-resource-per-operation Modulo regression
  set provides a direct comparison with Simplex:

  | problem | Simplex II / L | NodeRL II / L | CP-SAT II / L |
  | --- | ---: | ---: | ---: |
  | canis14_fig2 | 4 / 4 | 3 / 5 | 3 / 5 |
  | ceil_resource_mii | 3 / 3 | 3 / 3 | 3 / 3 |
  | minII_feasible | 3 / 14 | 3 / 14 | 3 / 14 |
  | minII_infeasible | 4 / 5 | 4 / 5 | 4 / 5 |
  | four_read_pipeline | 4 / 8 | 4 / 8 | 4 / 8 |

  The five instances take approximately 0.02 seconds together with either
  Simplex or NodeRL and 0.05 seconds with CP-SAT. NodeRL deliberately compares
  II before sink latency, explaining the canis tradeoff. Simplex remains the
  smaller deterministic choice in its supported domain. The held-resource
  proxies are not equivalent Simplex inputs; even after removing holds and
  multiple-resource uses, these generated recurrences currently report that
  the Simplex heuristic cannot update its frozen assignment while expanding
  II. That rejection is recorded as a failed run rather than compared as a
  timing result.

  Resource allocation is a separate outer decision. This command sweeps the
  two instance counts with costs 3 and 5 per multiplier and divider:

  ```sh
  python3 utils/run_modulo_node_rl_bench.py \
    --circt-opt build-fc092/bin/circt-opt --kernel 2mm \
    --chains 4 --unrolls 2 --episodes 32 --seeds 0 1 2 3 4 \
    --mul-limits 1 2 4 --mul-ii 3 --div-limits 1 2 --div-ii 2 \
    --mul-cost 3 --div-cost 5 --cpsat --timeout 10
  ```

  The CP-SAT non-dominated points and corresponding NodeRL medians are:

  | mul/div instances | cost | CP-SAT II / L | NodeRL II / median L |
  | ---: | ---: | ---: | ---: |
  | 1 / 1 | 8 | 48 / 48 | 48 / 48 |
  | 2 / 1 | 11 | 32 / 33 | 32 / 33 |
  | 2 / 2 | 16 | 24 / 24 | 24 / 42 |
  | 4 / 2 | 22 | 16 / 17 | 16 / 17 |

  At cost 16, a 32-node, one-second local search returns II/L=24/24 in all five
  seeds (0.275 seconds mean wall time). Thus the raw linear policy can miss a
  Pareto point through latency even when it finds every minimum II; the hybrid
  retains it without training a neural model. Across the full six-allocation
  sweep CP-SAT takes 0.172--0.531 seconds per point and NodeRL 0.028--0.040
  seconds per seed. Costs are scheduling surrogates, not synthesized area.

  Large held-resource graphs exercise exact resource/SCC decomposition and
  bulk virtual lanes. The following uses eight disjoint resource clusters, 32
  configured episodes, multiplier hold 3/limit 2, and divider hold 2/limit 1:

  ```sh
  python3 utils/run_modulo_node_rl_bench.py \
    --circt-opt build-fc092/bin/circt-opt --kernel 2mm \
    --chains 64 128 256 512 1024 --unrolls 2 --episodes 32 \
    --seed 7 --resource-clusters 8 \
    --mul-limit 2 --mul-ii 3 --div-limit 1 --div-ii 2
  ```

  | operations | NodeRL II / L | wall time |
  | ---: | ---: | ---: |
  | 513 | 64 / 65 | 0.035 s |
  | 1,025 | 128 / 129 | 0.053 s (ten-seed mean) |
  | 2,049 | 256 / 257 | 0.058 s |
  | 4,097 | 512 / 513 | 0.091 s |
  | 8,193 | 1024 / 1025 | 0.185 s |

  All reported runs attain the dominant resource-demand II bound and produce a
  static binding for every non-sink operation; the 1,025-node success rate is
  100% over ten seeds. These are scheduling-operation counts, not gates. They
  show that neither a GPU nor neural inference is needed at the current target
  scale. A single resource/SCC cluster containing thousands of mutually coupled
  operations can still be harder than this decomposable benchmark, so the work
  and ordering budgets remain necessary safeguards.

  Two structural regressions are especially relevant to interpreting the RL
  result. Signed resource offsets make NodeRL match CP-SAT on all five direct
  Simplex comparison cases, including starts separated by a whole II. For
  operations that simultaneously use two held resources, coupled bulk ordering
  changes 15- and 29-node synthetic results from II 13/26 to the exact II 12/24
  across ten seeds. These gains came from correcting the search space and its
  constraint representation, not from a larger policy.

  Neural inference is therefore gated rather than planned by default. Measure
  at least ten seeds on CP-SAT-solvable clusters and introduce a small
  GNN/actor-critic only if a median II or sink-latency gap above 10%, or a gap
  above 5% on at least three graph families, persists after structural fixes
  and a fixed local-search budget. Also require evidence that policy scoring,
  rather than constraint propagation, dominates runtime. The implemented MCTS
  is an opt-in, budgeted fixed-II feasibility rescue over small local states;
  global MCTS remains unsuitable because pairwise resource ordering has
  quadratic branching and conflict-proportional depth. If synthesis later
  supplies a black-box area/timing reward, a 20--100-node local tree whose
  actions select a resource order or LNS neighborhood may become useful.

  The present evidence does not cross that gate: raw Jacobi and Covariance
  latency gaps exceed 10%, but bounded local CP-SAT removes or nearly removes
  them, and the large CPU runs are already subsecond. The best episode's start
  times and bindings are committed together only after training; transient
  exploratory episodes never alter the selected schedule. Aggregate modulo
  reservations without static bindings still require the rotating arbitration
  described above. The modulo Simplex heuristic likewise reports failure if II
  expansion cannot update its frozen assignment or verification rejects its
  candidate, rather than emitting invalid IR.

- CP-SAT scheduler (requires OR-Tools):
  Solves `SharedOperatorsProblem` and `ModuloProblem` instances through
  `scheduler=cpsat`. For modulo instances it computes the resource-demand bound,
  raises it to the first II satisfying all cyclic difference constraints, then
  enumerates IIs in increasing order. At each candidate it exactly minimizes
  the designated sink start time and encodes every resource reservation phase.
  `report-statistics=true` prints both the lower bound and selected II. It is
  intended as a small-instance optimality oracle for NodeRL, not as the
  large-graph production scheduler. The low-level `scheduleCPSAT` call returns
  reservation feasibility and start times without static bindings. The
  `exploreCPSATPareto` wrapper turns each verified result into the same
  replayable certificate as NodeRL, synthesizing a bounded static coloring when
  possible and retaining rotating reservations otherwise.

- Integer linear programming-based scheduler
  ([`LPSchedulers.cpp`](https://github.com/llvm/circt/blob/main/lib/Scheduling/LPSchedulers.cpp)):
  Demo implementation for using an ILP solver via the OR-Tools integration.

### Integrating a scheduled implementation

SSP is a problem interchange format, rather than an implementation IR. An HLS
client should construct a `SharedOperatorsProblem` or `ModuloProblem` directly
from its source operations, invoke a scheduler or Pareto explorer, and read the
selected start times, bindings, or certificate when constructing its own
pipeline stages, control, and resource instances. For a modulo schedule it must
additionally lower the returned pipeline II and loop-carried values. The
client remains responsible for mapping an operator type to a typed
implementation primitive (including operand/result widths and a module or
Calyx primitive) and for connecting the assigned resource instance. This
separation lets the scheduling infrastructure remain dialect-agnostic while
preserving a path to RTL and post-synthesis cost calibration. A resource `cost`
is an inexpensive per-unit surrogate for exploration; it does not account for
allocation-dependent mux, control, register, or routing cost. Clients should
synthesize the small set of returned Pareto candidates, use the measured
area/timing/power to make the final choice, and, when appropriate, update their
resource-cost estimates for later explorations.

Keep three validation levels separate. A successful certificate replay proves
only modeled dependences, II, resource capacity, and assignment consistency.
Successful LoopSchedule/Calyx/HW lowering additionally proves that the selected
resource representation is structurally accepted by that target path. Area,
Fmax, power, routing congestion, and timing-closed throughput are PPA claims and
must come from logic synthesis and implementation; neither the linear resource
cost nor a verified schedule predicts them by itself.

For the existing Affine demonstration flow, pass
`-convert-affine-to-loopschedule="scheduler=node-rl"` selects Modulo NodeRL
instead of the default simplex scheduler; `scheduler=cpsat` selects the exact
reference solver in an OR-Tools-enabled build. Both lower the resulting
II/start times to `loopschedule.pipeline`; `node-rl-episodes`,
`node-rl-episode-node-budget`, `node-rl-resource-ordering-budget`, and
`node-rl-seed` control the reproducible NodeRL search.
`node-rl-mcts-trees`, `node-rl-mcts-simulations`, the tree/rollout widths, and
the MCTS time limit enable the otherwise-disabled fixed-II rescue.
`node-rl-local-search-nodes` and `node-rl-local-search-time-limit` enable the
optional OR-Tools neighborhood. With `multiplier-limit=N` and
`multiplier-ii=H`, the flow also models N shared multiplier instances with a
  hold/accept interval of H cycles and preserves static bindings as metadata.
  If CP-SAT returns only start times, the flow derives a periodic selector;
  selector period one is an ordinary static assignment.
For direct Affine CP-SAT experiments, `cpsat-time-limit`, `cpsat-workers`,
`cpsat-minimize-latency`, `cpsat-resource-model`, `cpsat-balanced-probe`, and
`cpsat-balanced-probe-time-limit` forward the corresponding exact-scheduler
controls. `cpsat-report-statistics` reports the proved lower bound, selected
II, effective resource model, and probe result.
`emit-ssp` prints the unscheduled problem produced by Affine dependence
analysis as a standalone SSP instance and skips LoopSchedule conversion. This
is intended for scheduler experiments that must round-trip through CIRCT's
strict problem verifier.
LoopSchedule-to-Calyx consumes a single-owner static binding and the
rotating selectors described above. It accepts cross-operation pools only when
their selected physical instance sets are disjoint; overlapping ownership
needs cycle-accurate stage control and completion tags. Merely placing
nominally time-disjoint groups in `calyx.par` would drive the same cell
concurrently. Other resource kinds still require target-specific lowering
support.

## Utilities

See
[`Utilities.h`](https://github.com/llvm/circt/blob/main/include/circt/Scheduling/Utilities.h):
- Topological graph traversal
- DFA to compute combinational path delays
- DOT dump

## Adding a new problem

*See e.g. [#2233](https://github.com/llvm/circt/pull/2233), which added the
`ChainingProblem`.*

- Decide where to add it. Guideline: If it is trait-like and similar to the
  existing problem mentioned above, add it to `Problems.h`. If the model is
  specific to your use-case, it is best to start out in locally in your
  dialect/pass.
- Declare the new problem class and inherit *virtually* from the relevant
  superclasses (at least `Problem`).
- Define additional properties (private), and the corresponding public
  getters/setters. Getters return `Optional<T>` values, to indicate an unset
  state.
   - Note that dependence properties are somewhat expensive to store, making it
     desirable that clients and algorithms expect and handle the unset state.
     This should be clearly documented. Example: `distance` property in
     `CyclicProblem`.
- Redefine the `getProperties(*)` methods to get dumping support. These should
  consider any properties the new class adds, plus properties defined in the
  superclass(es).
- Redefine `check()` (input constraints) and `verify()` (solution constraints).
  If possible, follow the
  [design used in the existing problem classes](#constraints).

### Testing

Please extend the [SSP](https://circt.llvm.org/docs/Dialects/SSP/) dialect to
enable testing of the new problem definition.
- If the problem defines any new properties, add them to
  [`SSPAttributes.td`](https://github.com/llvm/circt/blob/main/include/circt/Dialect/SSP/SSPAttributes.td).
- Instantiate the
  [`Default<ProblemT>`](https://github.com/llvm/circt/blob/main/include/circt/Dialect/SSP/Utilities.h#L457-L459)
  template for the new problem.
- Handle the problem class in the
  [`-ssp-roundtrip`](https://github.com/llvm/circt/blob/main/lib/Dialect/SSP/Transforms/Roundtrip.cpp)
  pass.
- Write a couple of "positive" testcases, as well as at least one error test for
  each input/solution constraint, as validated by `check()` / `verify()`. See
  the [existing test cases](https://github.com/llvm/circt/tree/main/test/Scheduling)
  for inspiration.

## Adding a new scheduler

*See e.g. [#2650](https://github.com/llvm/circt/pull/2650), which added a
scheduler for the `CyclicProblem`.*

- Schedulers should opt-in to specific problems by providing entry points for
  the problem subclasses they support. Example:
  ```c++
  LogicalResult awesomeScheduler(Problem &prob);
  LogicalResult awesomeScheduler(CyclicProblem &prob);
  ```
- Schedulers can expect that the input invariants were enforced by a
  `check()`-call in the client, and must compute a solution that complies with
  the solution constraints when the client calls the problem's `verify()`
  method.
- Schedulers can live anywhere. If a new algorithm is not entirely
  dialect/pass-specific and supports problems defined in `Problems.h`, it should
  offer entry points in `Algorithms.h`.
- Objectives are not part of the problem signature. Therefore, if an algorithm
  supports optimizing for different objectives, clients should be able to select
  one via the entry point(s).

### Testing

- To enable testing, add the new scheduler to the
  [`-ssp-schedule`](https://github.com/llvm/circt/blob/main/lib/Dialect/SSP/Transforms/Schedule.cpp)
  pass, and invoke it from the test cases for the supported problems
  ([example](https://github.com/llvm/circt/blob/main/test/Scheduling/problems.mlir#L2-L4)).
- If the algorithm may fail in certain situations (e.g., "linear program is
  infeasible"), add suitable error tests as well.
