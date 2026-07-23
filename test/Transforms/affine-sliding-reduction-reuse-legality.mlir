// RUN: circt-opt --affine-sliding-reduction-reuse %s > %t.once
// RUN: circt-opt --affine-sliding-reduction-reuse %t.once > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

// A source selected by the induction variable cannot be referenced by
// prologue loads. Its identity may differ between iterations.

// CHECK-LABEL: func.func @loop_local_source
// CHECK: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NOT: iter_args
// CHECK: %[[SOURCE:.*]] = arith.select
// CHECK-NEXT: %[[X0:.*]] = affine.load %[[SOURCE]][%[[I]]]
// CHECK-NEXT: %[[X1:.*]] = affine.load %[[SOURCE]][%[[I]] + 1]
func.func @loop_local_source(
    %a: memref<5xi32>, %b: memref<5xi32>) {
  %zero = arith.constant 0 : index
  affine.for %i = 0 to 4 {
    %useA = arith.cmpi eq, %i, %zero : index
    %source = arith.select %useA, %a, %b : memref<5xi32>
    %x0 = affine.load %source[%i] : memref<5xi32>
    %x1 = affine.load %source[%i + 1] : memref<5xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// Only leading-column loads become dead. The newest load remains available
// both to the rebuilt fold and to unrelated users in the original loop body.

// CHECK-LABEL: func.func @trailing_load_external_use
// CHECK: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][%[[LB]]]
// CHECK: affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1]
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[STATE]], %[[NEXT]] : i32
// CHECK-NEXT: %[[SIDE:.*]] = arith.subi %[[NEXT]], %{{.*}} : i32
// CHECK-NEXT: affine.store %[[SUM]]
// CHECK-NEXT: affine.store %[[SIDE]]
// CHECK-NEXT: affine.yield %[[NEXT]] : i32
func.func @trailing_load_external_use(
    %src0: memref<5xi32>, %out0: memref<4x2xi32>) {
  %src, %out = memref.distinct_objects %src0, %out0
      : memref<5xi32>, memref<4x2xi32>
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    %sum = arith.addi %x0, %x1 : i32
    %side = arith.subi %x1, %one : i32
    affine.store %sum, %out[%i, 0] : memref<4x2xi32>
    affine.store %side, %out[%i, 1] : memref<4x2xi32>
  }
  return
}

// A poison-sensitive root cannot be regrouped, but it does not hide a
// flagless load-only subtree that is independently safe to carry.

// CHECK-LABEL: func.func @nsw_root_safe_subtree
// CHECK: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][%[[LB]]]
// CHECK: affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1]
// CHECK-NEXT: %[[X2:.*]] = affine.load %[[SRC]][%[[I]] + 2]
// CHECK-NEXT: %[[SAFE:.*]] = arith.addi %[[STATE]], %[[NEXT]] : i32
// CHECK-NEXT: %[[ROOT:.*]] = arith.addi %[[SAFE]], %[[X2]]
// CHECK-SAME: overflow<nsw>
// CHECK-NEXT: affine.store %[[ROOT]]
// CHECK-NEXT: affine.yield %[[NEXT]] : i32
func.func @nsw_root_safe_subtree(
    %src0: memref<6xi32>, %out0: memref<4xi32>) {
  %src, %out = memref.distinct_objects %src0, %out0
      : memref<6xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i + 1] : memref<6xi32>
    %x2 = affine.load %src[%i + 2] : memref<6xi32>
    %safe = arith.addi %x0, %x1 : i32
    %root = arith.addi %safe, %x2 overflow<nsw> : i32
    affine.store %root, %out[%i] : memref<4xi32>
  }
  return
}

// Failure is candidate-local. A modified source at the start of the block
// must not prevent a later reduction over a proven-distinct source.

// CHECK-LABEL: func.func @modified_then_legal
// CHECK: %[[DISTINCT:.*]]:3 = memref.distinct_objects
// CHECK: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[PRELOAD:.*]] = affine.load %[[DISTINCT]]#1[%[[LB]]]
// CHECK: affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK-NEXT: %[[A0:.*]] = affine.load %[[DISTINCT]]#0[%[[I]]]
// CHECK-NEXT: %[[A1:.*]] = affine.load %[[DISTINCT]]#0[%[[I]] + 1]
// CHECK-NEXT: %[[BAD:.*]] = arith.addi %[[A0]], %[[A1]]
// CHECK-NEXT: affine.store %[[BAD]], %[[DISTINCT]]#0[%[[I]]]
// CHECK-NEXT: %[[BNEXT:.*]] = affine.load %[[DISTINCT]]#1[%[[I]] + 1]
// CHECK-NEXT: %[[GOOD:.*]] = arith.addi %[[STATE]], %[[BNEXT]]
// CHECK-NEXT: affine.store %[[GOOD]], %[[DISTINCT]]#2[%[[I]]]
// CHECK-NEXT: affine.yield %[[BNEXT]] : i32
func.func @modified_then_legal(
    %a0: memref<5xi32>, %b0: memref<5xi32>, %out0: memref<4xi32>) {
  %a, %b, %out = memref.distinct_objects %a0, %b0, %out0
      : memref<5xi32>, memref<5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0v = affine.load %a[%i] : memref<5xi32>
    %a1v = affine.load %a[%i + 1] : memref<5xi32>
    %bad = arith.addi %a0v, %a1v : i32
    affine.store %bad, %a[%i] : memref<5xi32>
    %b0v = affine.load %b[%i] : memref<5xi32>
    %b1v = affine.load %b[%i + 1] : memref<5xi32>
    %good = arith.addi %b0v, %b1v : i32
    affine.store %good, %out[%i] : memref<4xi32>
  }
  return
}
