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

// A loop iteration argument can also be forwarded into a different result
// position alongside a value computed in the loop body.

// CHECK-LABEL: func.func @forward_rotated
// CHECK: %[[ROTATED_STAGE:.+]]:3 = loopschedule.pipeline.stage start = 0
// CHECK: %[[SUM:.+]] = arith.addi
// CHECK: loopschedule.register %[[SUM]], {{.*}}, {{.*}} : i32, i32, index
// CHECK: loopschedule.terminator
// CHECK-SAME: iter_args(%[[ROTATED_STAGE]]#2, %[[ROTATED_STAGE]]#1,
// CHECK-SAME: %[[ROTATED_STAGE]]#0)
// CHECK-SAME: results(%[[ROTATED_STAGE]]#1, %[[ROTATED_STAGE]]#0)
func.func @forward_rotated(%lhs: i32, %rhs: i32) -> (i32, i32) {
  %result:2 = affine.for %i = 0 to 4
      iter_args(%a = %lhs, %b = %rhs) -> (i32, i32) {
    %sum = arith.addi %a, %b : i32
    affine.yield %b, %sum : i32, i32
  }
  return %result#0, %result#1 : i32, i32
}
