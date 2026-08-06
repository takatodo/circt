// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=1 multiplier-ii=1" -lower-loopschedule-to-calyx | FileCheck %s

// A single operation can own a static resource instance without introducing
// cross-group port contention.
// CHECK-LABEL: calyx.component @static_multiplier
// CHECK-COUNT-1: calyx.std_mult_pipe
func.func @static_multiplier(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg1 = 0 to 16 iter_args(%arg2 = %c0_i32) -> (i32) {
    %1 = arith.muli %arg2, %arg0 : i32
    affine.yield %1 : i32
  }
  return %0 : i32
}
