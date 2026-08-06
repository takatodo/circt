// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=1 multiplier-ii=1" -lower-loopschedule-to-calyx -verify-diagnostics

// The two multiplies are bound to the same NodeRL resource instance. Lowering
// currently places both stage groups in the steady-state calyx.par, losing
// their start=0/start=3 separation. Reject this until stage-time control can
// preserve the otherwise valid static sharing schedule.
func.func @shared_multiplier(%arg0: i32) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg1 = 0 to 16 iter_args(%arg2 = %c0_i32) -> (i32) {
    %1 = arith.muli %arg2, %arg0 : i32
    // expected-error@+1 {{multiple operations bound to static multiplier instance 'multiplier.0' require cycle-accurate stage control}}
    %2 = arith.muli %1, %arg0 : i32
    affine.yield %2 : i32
  }
  return %0 : i32
}
