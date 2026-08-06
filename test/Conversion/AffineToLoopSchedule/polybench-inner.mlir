// RUN: circt-opt %s -convert-affine-to-loopschedule="scheduler=node-rl multiplier-limit=2" | FileCheck %s

// CHECK-LABEL: func @gemm_inner
// CHECK: loopschedule.pipeline
// CHECK-LABEL: func @two_mm_inner
// CHECK: loopschedule.pipeline
// CHECK-LABEL: func @jacobi_2d_inner
// CHECK: loopschedule.pipeline
// CHECK-LABEL: func @covariance_inner
// CHECK: loopschedule.pipeline
func.func @gemm_inner(%a: memref<8x8xi32>, %b: memref<8x8xi32>,
                      %c: memref<8x8xi32>, %i: index, %j: index) {
  %zero = arith.constant 0 : i32
  %result = affine.for %k = 0 to 8 iter_args(%acc = %zero) -> (i32) {
    %av = affine.load %a[%i, %k] : memref<8x8xi32>
    %bv = affine.load %b[%k, %j] : memref<8x8xi32>
    %product = arith.muli %av, %bv : i32
    %sum = arith.addi %acc, %product : i32
    affine.yield %sum : i32
  }
  affine.store %result, %c[%i, %j] : memref<8x8xi32>
  return
}

func.func @two_mm_inner(%a: memref<8x8xi32>, %b: memref<8x8xi32>,
                        %c: memref<8x8xi32>, %d: memref<8x8xi32>,
                        %i: index, %j: index) {
  %zero = arith.constant 0 : i32
  %result = affine.for %k = 0 to 8 iter_args(%acc = %zero) -> (i32) {
    %av = affine.load %a[%i, %k] : memref<8x8xi32>
    %bv = affine.load %b[%k, %j] : memref<8x8xi32>
    %cv = affine.load %c[%k, %j] : memref<8x8xi32>
    %ab = arith.muli %av, %bv : i32
    %abc = arith.muli %ab, %cv : i32
    %sum = arith.addi %acc, %abc : i32
    affine.yield %sum : i32
  }
  affine.store %result, %d[%i, %j] : memref<8x8xi32>
  return
}

func.func @jacobi_2d_inner(%a: memref<10x10xi32>, %b: memref<10x10xi32>,
                           %i: index) {
  affine.for %j = 1 to 9 {
    %north = affine.load %a[%i - 1, %j] : memref<10x10xi32>
    %south = affine.load %a[%i + 1, %j] : memref<10x10xi32>
    %west = affine.load %a[%i, %j - 1] : memref<10x10xi32>
    %east = affine.load %a[%i, %j + 1] : memref<10x10xi32>
    %pair0 = arith.addi %north, %south : i32
    %pair1 = arith.addi %west, %east : i32
    %sum = arith.addi %pair0, %pair1 : i32
    affine.store %sum, %b[%i, %j] : memref<10x10xi32>
  }
  return
}

func.func @covariance_inner(%data: memref<8x8xi32>, %cov: memref<8x8xi32>,
                            %i: index, %j: index) {
  %zero = arith.constant 0 : i32
  %result = affine.for %k = 0 to 8 iter_args(%acc = %zero) -> (i32) {
    %left = affine.load %data[%k, %i] : memref<8x8xi32>
    %right = affine.load %data[%k, %j] : memref<8x8xi32>
    %product = arith.muli %left, %right : i32
    %sum = arith.addi %acc, %product : i32
    affine.yield %sum : i32
  }
  affine.store %result, %cov[%i, %j] : memref<8x8xi32>
  return
}
