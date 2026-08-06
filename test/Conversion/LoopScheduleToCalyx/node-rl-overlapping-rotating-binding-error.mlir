// RUN: circt-opt %s -lower-loopschedule-to-calyx -verify-diagnostics

// Both source operations select the same physical instances. A done pulse
// cannot identify which destination register owns the result, so this remains
// rejected until request tags are delayed alongside the primitive pipeline.
func.func @overlapping_rotating_multipliers(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %0:2 = loopschedule.pipeline II = 3 trip_count = 4 iter_args(%arg2 = %c0, %arg3 = %c0_i32, %arg4 = %c0_i32) : (index, i32, i32) -> (i32, i32) {
    %1 = arith.cmpi ult, %arg2, %c4 : index
    loopschedule.register %1 : i1
  } do {
    %1:3 = loopschedule.pipeline.stage start = 0 {
      %2 = arith.muli %arg3, %arg0 {circt.rotating_resource_reservations = [{hold = 4 : i64, instances = 2 : i64, period = 3 : i64, phase = 0 : i64, resource = "multiplier", selector = [0, 1], selector_period = 2 : i64}]} : i32
      // expected-error@+1 {{rotating multiplier instance 'multiplier.0' is selected by multiple operations and requires completion ownership tracking}}
      %3 = arith.muli %arg4, %arg1 {circt.rotating_resource_reservations = [{hold = 4 : i64, instances = 2 : i64, period = 3 : i64, phase = 0 : i64, resource = "multiplier", selector = [0, 1], selector_period = 2 : i64}]} : i32
      %4 = arith.addi %arg2, %c1 : index
      loopschedule.register %2, %3, %4 : i32, i32, index
    } : i32, i32, index
    loopschedule.terminator iter_args(%1#2, %1#0, %1#1), results(%1#0, %1#1) : (index, i32, i32) -> (i32, i32)
  }
  return %0#0, %0#1 : i32, i32
}
