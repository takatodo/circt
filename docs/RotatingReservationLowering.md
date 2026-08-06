# Rotating Reservation Lowering

## Status

Modulo NodeRL can schedule a resource whose hold time is greater than the
pipeline II, while CP-SAT can return a feasible periodic schedule without any
instance bindings. `circt.rotating_resource_reservations` preserves an
implementable instance-selection table for both cases on the corresponding
LoopSchedule operation. It is a target-neutral contract.
LoopSchedule-to-Calyx now implements a deliberately constrained slice of that
contract: rotating integer multiply operations whose selected physical
instance sets are disjoint. Multiple operations may use one resource pool when
the joint selector coloring partitions that pool between them.

## Selector-table contract

An entry contains `resource`, `phase`, `period`, `hold`, and `instances`.
When periodic coloring succeeds it also contains `selector` and
`selector_period`. `selector[k]` names the physical instance to issue for
iteration class `k mod selector_period`.
A selector period of one is a static allocation synthesized for an unbound
start-time schedule; a longer selector rotates between iterations.

The table generator colors reservation intervals on a finite cyclic horizon,
including wraparound. It searches selector periods from one through the
resource limit and keeps the shortest feasible table. `instances` records the
number of colors actually used, not the available upper bound, so a limit of
four with selector `[0, 1]` materializes only two cells. The generator currently
requires all uses of a resource to be unbound; mixing pre-existing static
bindings with synthesized selectors needs a joint allocator.
Affine-to-LoopSchedule emits reservation metadata only when this coloring
succeeds; an operation without a static binding is not by itself a realizable
allocation.

## Calyx boundary

LoopSchedule-to-Calyx creates only the `std_mult_pipe` instances referenced by
the selector tables. For a selector longer than one it also creates a phase
register, resets that register before each pipeline invocation, compares it
against every selector class, conditionally drives only the selected instance,
joins that operation's instance completion signals into its result register,
and advances the phase only when the selected operation completes. A
period-one selector directly guards its single selected instance and requires
no phase register.

This is enough for the representative single-stage loop-carried recurrence:
pipeline II 3, resource hold 4, and selector `[0, 1]`. It is not a general
cycle-accurate pipeline lowering. The existing lowering emits stage groups in
a steady-state `calyx.par` and does not encode distinct stage start times.
Consequently it rejects all of the following rather than generating port
contention or an incorrect resource count:

- two distinct operations selecting the same rotating physical instance,
  because completion ownership must be delayed and routed along with each
  request;
- two distinct operations bound to one static multiplier instance, even when
  `hold == 1`, because their scheduled start-cycle separation is lost; and
- any static multiplier binding with `hold > 1`.

A single operation that owns a static `hold == 1` instance remains supported.

General rotating lowering still requires:

1. a selector register updated once per logical pipeline iteration;
2. a combinational comparison against the selector class;
3. conditional enable of the selected physical instance for each stage; and
4. a result path that joins the selected instance's completion to the stage
   register.

The same control substrate must represent stage start time and pipeline II
exactly. Only then can multi-operation static and rotating resource counts be
compared at RTL or synthesis level. Automated regression coverage stops at
verified Calyx IR. The representative case also passes through the installed
native Calyx compiler and `lower-calyx-to-hw`, producing two `comb.mul`
operations and one phase register.

A separate CP-SAT representative uses two simultaneous multiplications,
multiplier hold three, pipeline II three, and a two-instance limit. The start
times are colored as period-one selectors `[0]` and `[1]`. Regression tests
verify the two-cell Calyx pool, and a manual run through `calyx-native`,
`lower-calyx-to-hw`, `lower-seq-to-sv`, and Verilog export produces exactly two
RTL multipliers and no selector-phase register. This remains a structural
result because of the native-lowering register protocol described below.

An additional five-iteration recurrence experiment initializes its carried
value to one and multiplies by three, so the expected result is 243. Exporting
the generated Calyx through `circt-translate --export-calyx`, compiling it
with the Rust Calyx 0.7.1 Verilog backend, and simulating with Verilator 5.044
passes for both a one-instance `hold=1` control case (30 cycles) and the
two-instance rotating `hold=4` case (43 cycles). These cycle counts are not a
performance comparison because the resource hold constraints differ; the
experiment establishes functional selector, phase-update, and result-routing
behavior.

Passing the same two Calyx programs through CIRCT's `calyx-native` followed by
`lower-calyx-to-hw` produces structurally valid RTL, but both simulations time
out. This is independent of rotating selection: the generated control FSM is a
`calyx.register` with continuously asserted `write_en`, while
`CalyxToHW.cpp` currently gates its data clock-enable with
`write_en && !done`. Its registered `done` stays asserted after the first
cycle, so the FSM advances only once. The direct Rust backend implements the
Calyx register semantics and completes both programs. Until that pre-existing
native-lowering mismatch is resolved, functional validation uses the direct
Calyx Verilog path and CIRCT's HW path is structural only. No Yosys or
post-placement measurements are claimed.

## Representative resource/II experiment

For the one-multiply recurrence in `node-rl-rotating-selector.mlir`, with a
resource hold of four cycles:

| multiplier limit | scheduled II | emitted allocation | HW result |
| ---: | ---: | --- | --- |
| 1 | 4 | one static `hold=4` binding | rejected at Calyx safety boundary |
| 2 | 3 | `selector=[0,1]`, two instances | two `comb.mul` operations |
| 4 | 3 | compressed to the same two-instance selector | two `comb.mul` operations |

Thus the representative rotating point spends one additional multiplier over
the one-instance schedule and changes steady-state initiation rate from `1/4`
to `1/3` (about 33% higher). Increasing the upper bound from two to four no
longer over-allocates hardware.

## Regression coverage

- `node-rl-rotating-selector.mlir` verifies `selector<[0, 1]>` generation.
- `node-rl-rotating-binding-error.mlir` verifies the single-operation Calyx
  lowering: two multiplier cells, phase initialization, guarded dispatch, and
  completion-driven phase advance.
- `node-rl-multiple-rotating-binding.mlir` verifies two operations with
  disjoint NodeRL selector sets (`[0,1]` and `[2,3]`), plus CP-SAT period-one
  selectors `[0]` and `[1]` and the resulting two-cell Calyx pool.
- `cpsat-options.mlir` verifies that an unbound CP-SAT schedule retains its
  period-one multiplier selectors in LoopSchedule IR.
- `node-rl-overlapping-rotating-binding-error.mlir` verifies rejection when
  two operations select the same physical instances.
- `node-rl-nonpipelined-binding-error.mlir` verifies rejection of static
  `hold > 1` bindings.
- `node-rl-shared-multiplier.mlir` verifies rejection of unsafe cross-stage
  static sharing.
- `node-rl-static-multiplier.mlir` verifies a single static owner.
