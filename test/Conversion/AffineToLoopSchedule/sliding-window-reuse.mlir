// RUN: circt-opt --affine-sliding-window-reuse --convert-affine-to-loopschedule %s | FileCheck %s

// Sliding-window state introduced as affine.for iter_args must remain a
// rotating pair of pipeline iteration arguments. The loop body retains only
// the new tail load.

// CHECK-LABEL: func.func @sliding_window_to_pipeline
// CHECK: %[[X0:.*]] = memref.load %[[SRC:.*]][%{{.*}}] : memref<10xi32>
// CHECK: %[[X1:.*]] = memref.load %[[SRC]][%{{.*}}] : memref<10xi32>
// CHECK: %{{.*}}:2 = loopschedule.pipeline II =  1 trip_count =  8
// CHECK-SAME: iter_args(%[[IV:.*]] = %{{.*}}, %[[S0:.*]] = %[[X0]], %[[S1:.*]] = %[[X1]])
// CHECK: %[[STAGE0:.*]]:6 = loopschedule.pipeline.stage start = 0 {
// CHECK: %[[NEXT:.*]] = memref.load %[[SRC]][%{{.*}}] : memref<10xi32>
// CHECK: loopschedule.register {{.*}}, {{.*}}, %[[NEXT]], {{.*}}, %[[S1]], {{.*}} : index, index, i32, i32, i32, index
// CHECK: %[[STAGE1:.*]]:2 = loopschedule.pipeline.stage start = 1 {
// CHECK: loopschedule.register %[[STAGE0]]#2, {{.*}} : i32, i32
// CHECK: loopschedule.terminator iter_args(%[[STAGE0]]#5, %[[STAGE0]]#4, %[[STAGE1]]#0), results(%[[STAGE0]]#4, %[[STAGE1]]#0)
func.func @sliding_window_to_pipeline(%src: memref<10xi32>,
                                      %dst: memref<8xi32>) {
  %src_distinct, %dst_distinct = memref.distinct_objects %src, %dst
    : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src_distinct[%i] : memref<10xi32>
    %x1 = affine.load %src_distinct[%i + 1] : memref<10xi32>
    %x2 = affine.load %src_distinct[%i + 2] : memref<10xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %dst_distinct[%i] : memref<8xi32>
  }
  return
}
