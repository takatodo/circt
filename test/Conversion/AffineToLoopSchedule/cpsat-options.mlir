// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=3 cpsat-time-limit=2 cpsat-workers=1 cpsat-minimize-latency=false cpsat-balanced-probe=false cpsat-balanced-probe-time-limit=0.2 cpsat-resource-model=cumulative cpsat-report-statistics=true" 2>&1 | FileCheck %s %}
// RUN: %if or-tools %{ not circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat cpsat-resource-model=invalid" 2>&1 | FileCheck %s --check-prefix=BAD-MODEL %}

// CHECK: affine-cpsat: status=optimal, lower-bound=3, II=3, resource-model=cumulative
// CHECK-LABEL: func @dot_mul_accumulate
// CHECK: loopschedule.pipeline II = 3
// CHECK: arith.muli
// CHECK-SAME: resource = "multiplier"
// CHECK-SAME: selector = [0]
// CHECK-SAME: selector_period = 1
// CHECK: arith.muli
// CHECK-SAME: resource = "multiplier"
// CHECK-SAME: selector = [1]
// CHECK-SAME: selector_period = 1
// BAD-MODEL: error: invalid cpsat-resource-model; expected auto, onehot, or cumulative
func.func @dot_mul_accumulate(%arg0: memref<64xi32>,
                              %arg1: memref<64xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %index = 0 to 64
      iter_args(%accumulator = %zero) -> (i32) {
    %lhs = affine.load %arg0[%index] : memref<64xi32>
    %rhs = affine.load %arg1[%index] : memref<64xi32>
    %product = arith.muli %lhs, %rhs : i32
    %next = arith.muli %accumulator, %product : i32
    affine.yield %next : i32
  }
  return %result : i32
}
