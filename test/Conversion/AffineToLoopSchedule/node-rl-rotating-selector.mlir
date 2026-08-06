// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=2 multiplier-ii=4" | FileCheck %s
// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=4 multiplier-ii=4" | FileCheck %s --check-prefix=MINIMAL

// CHECK: circt.rotating_resource_reservations = [{hold = 4 : i64, instances = 2 : i64, period = 3 : i64, phase = 0 : i64, resource = "multiplier", selector = [0, 1], selector_period = 2 : i64}]
// MINIMAL: circt.rotating_resource_reservations = [{hold = 4 : i64, instances = 2 : i64, period = 3 : i64, phase = 0 : i64, resource = "multiplier", selector = [0, 1], selector_period = 2 : i64}]
func.func @rotating_multiplier_selector(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg1 = 0 to 16 iter_args(%arg2 = %c0_i32) -> (i32) {
    %1 = arith.muli %arg2, %arg0 : i32
    affine.yield %1 : i32
  }
  return %0 : i32
}
