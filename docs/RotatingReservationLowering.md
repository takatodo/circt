# Rotating Reservation Lowering

## Status

Modulo schedulers such as CP-SAT can return a feasible periodic start-time
schedule without assigning physical resource instances.
`circt.rotating_resource_reservations` preserves an implementable
instance-selection table on the corresponding LoopSchedule operation. It is a
target-neutral contract. LoopSchedule-to-Calyx implements a deliberately
constrained slice for pipelined integer multipliers.

## Selector-table contract

An entry contains `resource`, `phase`, `period`, `hold`, and `instances`. When
periodic coloring succeeds it also contains `selector` and `selector_period`.
`selector[k]` names the physical instance used by iteration class
`k mod selector_period`. A selector period of one is a static allocation; a
longer selector rotates between iterations.

The table generator colors reservation intervals on a finite cyclic horizon,
including wraparound. It searches selector periods from one through the
resource limit and keeps the shortest feasible table. `instances` records the
number of colors actually used rather than the configured upper bound, so a
limit of four with selector `[0, 1]` materializes only two cells. All uses of a
resource must currently be unbound; mixing an existing static binding with a
synthesized selector requires a joint allocator.

Affine-to-LoopSchedule emits this metadata only after the schedule has passed
the scheduling problem verifier and the periodic coloring succeeds.

## Calyx boundary

LoopSchedule-to-Calyx creates only the `std_mult_pipe` instances referenced by
the selector. For a selector longer than one it also creates a phase register,
resets it before each pipeline invocation, conditionally drives the selected
instance, joins completion into the result register, and advances the phase on
completion. A period-one selector uses its selected instance directly and
needs no phase register.

The current lowering does not encode arbitrary stage start times in its
steady-state `calyx.par`. It therefore rejects these cases rather than
generating unsafe port contention:

- distinct operations selecting the same rotating instance;
- distinct operations bound to the same static multiplier instance; and
- a static multiplier binding with `hold > 1`.

A single static `hold == 1` owner is supported. General sharing additionally
needs request ownership to travel with completion and cycle-accurate stage
control.

## Validation boundary

A one-operation recurrence with pipeline II 3, resource hold 4, and limit 2
produces selector `[0, 1]`, two Calyx multiplier cells, and a phase register.
A separate CP-SAT schedule with two simultaneous multiplications, hold 3, and
limit 2 produces period-one selectors `[0]` and `[1]`, two multiplier cells,
and no phase register.

The latter also passes through `calyx-native`, `lower-calyx-to-hw`,
`lower-seq-to-sv`, and Verilog export with exactly two RTL multipliers. This is
a structural result. Area, frequency, and power remain synthesis-level
measurements and are not implied by schedule verification.

Regression coverage is in `cpsat-options.mlir` and
`cpsat-resource-lowering.mlir`.
