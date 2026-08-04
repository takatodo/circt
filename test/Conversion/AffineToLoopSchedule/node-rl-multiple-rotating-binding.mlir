// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=4 multiplier-ii=4" -lower-loopschedule-to-calyx | FileCheck %s

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
func.func @multiple_rotating_multipliers(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = affine.for %arg2 = 0 to 16 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
    %1 = arith.muli %arg3, %arg0 : i32
    %2 = arith.muli %arg4, %arg1 : i32
    affine.yield %1, %2 : i32, i32
  }
  return %0#0, %0#1 : i32, i32
}
