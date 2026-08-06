// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=4 multiplier-ii=4" -lower-loopschedule-to-calyx | FileCheck %s
// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=3 cpsat-time-limit=2 cpsat-workers=1 cpsat-resource-model=cumulative" | FileCheck %s --check-prefix=CPSAT-SCHEDULE %}
// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=3 cpsat-time-limit=2 cpsat-workers=1 cpsat-resource-model=cumulative" -lower-loopschedule-to-calyx | FileCheck %s --check-prefix=CPSAT-CALYX %}

// The joint selector coloring partitions the physical pool: the first source
// operation owns instances 0/1 and the second owns 2/3. Distinct completion
// ports make this multi-operation case safe without an ownership tag.
// CHECK-LABEL: calyx.component @multiple_rotating_multipliers
// CHECK-DAG: calyx.register @rotating_phase_0_reg
// CHECK-DAG: calyx.register @rotating_phase_1_reg
// CHECK-DAG: calyx.std_mult_pipe @std_mult_pipe_0
// CHECK-DAG: calyx.std_mult_pipe @std_mult_pipe_1
// CHECK-DAG: calyx.std_mult_pipe @std_mult_pipe_2
// CHECK-DAG: calyx.std_mult_pipe @std_mult_pipe_3
// CHECK-DAG: calyx.assign %std_mult_pipe_0.left = %std_eq_0.out ?
// CHECK-DAG: calyx.assign %std_mult_pipe_1.left = %std_eq_1.out ?
// CHECK-DAG: calyx.assign %std_mult_pipe_2.left = %std_eq_2.out ?
// CHECK-DAG: calyx.assign %std_mult_pipe_3.left = %std_eq_3.out ?

// CP-SAT returns start times without bindings. The Affine lowering must derive
// a period-one selector which assigns the two simultaneous operations to two
// distinct instances, and Calyx must realize exactly that two-cell pool.
// CPSAT-SCHEDULE: loopschedule.pipeline II = 3
// CPSAT-SCHEDULE-DAG: resource = "multiplier", selector = [0], selector_period = 1
// CPSAT-SCHEDULE-DAG: resource = "multiplier", selector = [1], selector_period = 1
// CPSAT-CALYX-LABEL: calyx.component @multiple_rotating_multipliers
// CPSAT-CALYX-DAG: calyx.std_mult_pipe @std_mult_pipe_0
// CPSAT-CALYX-DAG: calyx.std_mult_pipe @std_mult_pipe_1
// CPSAT-CALYX-NOT: calyx.std_mult_pipe @std_mult_pipe_2
// CPSAT-CALYX-NOT: rotating_phase
func.func @multiple_rotating_multipliers(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
    %1 = arith.muli %arg3, %arg0 : i32
    %2 = arith.muli %arg4, %arg1 : i32
    affine.yield %1, %2 : i32, i32
  }
  return %0#0, %0#1 : i32, i32
}
