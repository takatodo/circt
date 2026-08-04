// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=2 multiplier-ii=4" -lower-loopschedule-to-calyx | FileCheck %s
// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=4 multiplier-ii=4" -lower-loopschedule-to-calyx | FileCheck %s --check-prefix=MINIMAL

// The recurrence needs II=3, while a multiplier request is held for four
// cycles. The phase alternates between two physical multipliers. This first
// lowering slice routes one operation over two physical instances; disjoint
// multi-operation pools and overlapping ownership are tested separately.
// CHECK-LABEL: calyx.component @rotating_multiplier
// CHECK: calyx.register @rotating_phase_0_reg
// CHECK-COUNT-2: calyx.std_mult_pipe
// CHECK: calyx.group @bb0_1 {
// CHECK: calyx.assign %std_mult_pipe_0.left = %std_eq_0.out ?
// CHECK: calyx.assign %std_mult_pipe_1.left = %std_eq_1.out ?
// CHECK: calyx.assign %rotating_phase_0_reg.write_en = %std_or_0.out
// CHECK: calyx.group @rotating_phase_init_0 {
// CHECK: calyx.assign %rotating_phase_0_reg.in = %false
// CHECK: calyx.enable @rotating_phase_init_0
// MINIMAL-LABEL: calyx.component @rotating_multiplier
// MINIMAL-COUNT-2: calyx.std_mult_pipe
func.func @rotating_multiplier(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg1 = 0 to 16 iter_args(%arg2 = %c0_i32) -> (i32) {
    %1 = arith.muli %arg2, %arg0 : i32
    affine.yield %1 : i32
  }
  return %0 : i32
}
