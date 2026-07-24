// RUN: circt-opt --convert-affine-to-loopschedule %s | FileCheck %s

// CHECK-LABEL: func.func @unresolved_read_read
// CHECK: loopschedule.pipeline
func.func @unresolved_read_read(
    %arg0: memref<3xi32>, %condition: i1) {
  %out = memref.alloca() : memref<3xi32>
  affine.for %i = 1 to 3 {
    %current = affine.load %arg0[%i] : memref<3xi32>
    %previous = scf.if %condition -> i32 {
      %value = affine.load %arg0[%i - 1] : memref<3xi32>
      scf.yield %value : i32
    } else {
      scf.yield %current : i32
    }
    %sum = arith.addi %current, %previous : i32
    affine.store %sum, %out[%i] : memref<3xi32>
  }
  return
}
