// RUN: circt-opt --affine-sliding-reduction-reuse --convert-affine-to-loopschedule %s | FileCheck %s

// The reduction-reuse transform introduces two carried partial sums.  The
// conversion must forward those values through pipeline stages and preserve
// the II=1 schedule.

// CHECK-LABEL: func.func @box3_i32
// CHECK: %{{.*}}:2 = loopschedule.pipeline II =  1 trip_count =  8 iter_args
// CHECK: %{{.*}}:6 = loopschedule.pipeline.stage start = 0
// CHECK: %{{.*}}:2 = loopschedule.pipeline.stage start = 1
// CHECK: loopschedule.terminator iter_args
// CHECK-SAME: results
func.func @box3_i32(%src0: memref<16xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<16xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<16xi32>
    %x1 = affine.load %src[%i + 1] : memref<16xi32>
    %x2 = affine.load %src[%i + 2] : memref<16xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %dst[%i] : memref<8xi32>
  }
  return
}
