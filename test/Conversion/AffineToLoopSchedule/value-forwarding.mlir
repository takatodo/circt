// RUN: circt-opt --convert-affine-to-loopschedule %s | FileCheck %s

// A loop-carried value may be yielded unchanged. The materialized pipeline
// still has to register that value in a stage before using it as the next
// iteration's argument and as a pipeline result.

// CHECK-LABEL: func.func @forward_unchanged
// CHECK: loopschedule.pipeline II =  1
// CHECK: %[[STAGE:.+]]:2 = loopschedule.pipeline.stage start = 0
// CHECK: loopschedule.register {{.*}}, {{.*}} : i32, index
// CHECK: loopschedule.terminator
// CHECK-SAME: iter_args(%[[STAGE]]#1, %[[STAGE]]#0)
// CHECK-SAME: results(%[[STAGE]]#0)
func.func @forward_unchanged(%dst: memref<4xi32>, %init: i32) -> i32 {
  %result = affine.for %i = 0 to 4
      iter_args(%value = %init) -> i32 {
    affine.store %value, %dst[%i] : memref<4xi32>
    affine.yield %value : i32
  }
  return %result : i32
}

// Forwarding a loop argument to the pipeline terminator is independent of
// whether later scheduled stages exist. In particular, this must not require
// materializing forwarding-only stages for the latency gap.

// CHECK-LABEL: func.func @forward_across_scheduled_stages
// CHECK: %[[FIRST:.+]]:3 = loopschedule.pipeline.stage start = 0
// CHECK: arith.muli
// CHECK: loopschedule.pipeline.stage start = 3
// CHECK: memref.store
// CHECK: loopschedule.terminator
// CHECK-SAME: iter_args(%[[FIRST]]#2, %[[FIRST]]#1)
// CHECK-SAME: results(%[[FIRST]]#1)
func.func @forward_across_scheduled_stages(
    %dst: memref<4xi32>, %init: i32, %factor: i32) -> i32 {
  %result = affine.for %i = 0 to 4
      iter_args(%value = %init) -> i32 {
    %long = arith.muli %value, %factor : i32
    affine.store %long, %dst[%i] : memref<4xi32>
    affine.yield %value : i32
  }
  return %result : i32
}

// Multiple loop iteration arguments can be forwarded into different result
// positions alongside an ordinary value computed in the loop body. The
// ordinary result needs zero forwarding stages; each block argument needs
// exactly one stage result. Neither case should create a new time slot.

// CHECK-LABEL: func.func @forward_rotated
// CHECK: %[[ROTATED_STAGE:.+]]:4 = loopschedule.pipeline.stage start = 0
// CHECK: %[[SUM:.+]] = arith.addi
// CHECK: loopschedule.register %[[SUM]], {{.*}}, {{.*}}, {{.*}} :
// CHECK-SAME: i32, i32, i32, index
// CHECK-NOT: loopschedule.pipeline.stage start = 1
// CHECK: loopschedule.terminator
// CHECK-SAME: iter_args(%[[ROTATED_STAGE]]#3, %[[ROTATED_STAGE]]#1,
// CHECK-SAME: %[[ROTATED_STAGE]]#2, %[[ROTATED_STAGE]]#0)
// CHECK-SAME: results(%[[ROTATED_STAGE]]#1, %[[ROTATED_STAGE]]#2,
// CHECK-SAME: %[[ROTATED_STAGE]]#0)
func.func @forward_rotated(%lhs: i32, %rhs: i32, %seed: i32)
    -> (i32, i32, i32) {
  %result:3 = affine.for %i = 0 to 4
      iter_args(%a = %lhs, %b = %rhs, %acc = %seed) -> (i32, i32, i32) {
    %sum = arith.addi %a, %b : i32
    affine.yield %b, %a, %sum : i32, i32, i32
  }
  return %result#0, %result#1, %result#2 : i32, i32, i32
}
