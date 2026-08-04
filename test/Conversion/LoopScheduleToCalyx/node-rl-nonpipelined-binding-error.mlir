// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=1 multiplier-ii=2" -lower-loopschedule-to-calyx -verify-diagnostics

// A binding with a resource hold greater than one needs cycle-accurate
// pipeline control. Do not lower it as an unconstrained Calyx parallel group.
func.func @nonpipelined_shared_multiplier(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg1 = 0 to 16 iter_args(%arg2 = %c0_i32) -> (i32) {
    // expected-error@+1 {{cannot lower a non-fully-pipelined multiplier binding to Calyx without cycle-accurate pipeline control}}
    %1 = arith.muli %arg2, %arg0 : i32
    %2 = arith.muli %1, %arg0 : i32
    affine.yield %2 : i32
  }
  return %0 : i32
}
