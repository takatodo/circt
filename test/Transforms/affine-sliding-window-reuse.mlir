// RUN: circt-opt --affine-sliding-window-reuse %s > %t.once
// RUN: circt-opt --affine-sliding-window-reuse --affine-sliding-window-reuse %s > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

// CHECK-LABEL: func.func @three_tap
// CHECK: %[[DISTINCT:.*]]:2 = memref.distinct_objects
// CHECK-NEXT: %[[X0:.*]] = affine.load %[[DISTINCT]]#0[0] {test.marker = "x0"} : memref<10xi32>
// CHECK-NEXT: %[[X1:.*]] = affine.load %[[DISTINCT]]#0[1] {test.marker = "x1"} : memref<10xi32>
// CHECK-NEXT: %{{.*}}:2 = affine.for %[[I:.*]] = 0 to 8
// CHECK-SAME: iter_args(%[[S0:.*]] = %[[X0]], %[[S1:.*]] = %[[X1]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[DISTINCT]]#0[%[[I]] + 2] : memref<10xi32>
// CHECK-NEXT: %[[SUM0:.*]] = arith.addi %[[S0]], %[[S1]] : i32
// CHECK-NEXT: %[[SUM1:.*]] = arith.addi %[[SUM0]], %[[NEXT]] : i32
// CHECK-NEXT: affine.store %[[SUM1]], %[[DISTINCT]]#1[%[[I]]] : memref<8xi32>
// CHECK-NEXT: affine.yield %[[S1]], %[[NEXT]] : i32, i32
// CHECK-NEXT: } {test.marker = "loop"}
func.func @three_tap(%src: memref<10xi32>, %dst: memref<8xi32>) {
  %src_distinct, %dst_distinct = memref.distinct_objects %src, %dst
    : memref<10xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src_distinct[%i] {test.marker = "x0"} : memref<10xi32>
    %x1 = affine.load %src_distinct[%i + 1] {test.marker = "x1"} : memref<10xi32>
    %x2 = affine.load %src_distinct[%i + 2] : memref<10xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %dst_distinct[%i] : memref<8xi32>
  } {test.marker = "loop"}
  return
}

// A window may start at any constant offset. The preloads are relative to the
// loop's lower bound, not necessarily absolute indices zero and one.

// CHECK-LABEL: func.func @nonzero_start
// CHECK: %[[X4:.*]] = affine.load %[[SRC:.*]][4] : memref<10xi32>
// CHECK-NEXT: %[[X5:.*]] = affine.load %[[SRC]][5] : memref<10xi32>
// CHECK: %{{.*}}:2 = affine.for %[[I:.*]] = 1 to 5
// CHECK-SAME: iter_args(%[[S3:.*]] = %[[X4]], %[[S4:.*]] = %[[X5]])
// CHECK: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 5] : memref<10xi32>
// CHECK: %[[SUM:.*]] = arith.addi %[[S3]], %[[S4]] : i32
// CHECK: %[[SUM2:.*]] = arith.addi %[[SUM]], %[[NEXT]] : i32
// CHECK: affine.yield %[[S4]], %[[NEXT]] : i32, i32
func.func @nonzero_start(%src: memref<10xi32>) {
  affine.for %i = 1 to 5 {
    %x3 = affine.load %src[%i + 3] : memref<10xi32>
    %x4 = affine.load %src[%i + 4] : memref<10xi32>
    %x5 = affine.load %src[%i + 5] : memref<10xi32>
    %sum0 = arith.addi %x3, %x4 : i32
    %sum1 = arith.addi %sum0, %x5 : i32
  }
  return
}

// Existing iter_args and externally used results stay first; window state is
// appended. Replacing a load rewrites all of its uses, not just a recognized
// arithmetic chain.

// CHECK-LABEL: func.func @existing_iter_args_and_multiple_uses
// CHECK: %[[INIT:.*]] = arith.constant 0 : i32
// CHECK: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][0] : memref<5xi32>
// CHECK: %[[RESULT:.*]]:2 = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[ACC:.*]] = %[[INIT]], %[[STATE:.*]] = %[[PRELOAD]])
// CHECK: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1] : memref<5xi32>
// CHECK: %[[TWICE:.*]] = arith.addi %[[STATE]], %[[STATE]] : i32
// CHECK: %[[TOTAL:.*]] = arith.addi %[[ACC]], %[[TWICE]] : i32
// CHECK: %[[UPDATED:.*]] = arith.addi %[[TOTAL]], %[[NEXT]] : i32
// CHECK: affine.yield %[[UPDATED]], %[[NEXT]] : i32, i32
// CHECK: return %[[RESULT]]#0 : i32
func.func @existing_iter_args_and_multiple_uses(%src: memref<5xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %i = 0 to 4 iter_args(%acc = %zero) -> i32 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    %twice = arith.addi %x0, %x0 : i32
    %total = arith.addi %acc, %twice : i32
    %updated = arith.addi %total, %x1 : i32
    affine.yield %updated : i32
  }
  return %result : i32
}

#dynamic_upper = affine_map<()[s0] -> (s0 + 4)>
#decreasing_upper = affine_map<()[s0] -> (s0 - 1)>
#overflowing_lower = affine_map<()[s0] -> (s0 + 9223372036854775806)>
#overflowing_upper = affine_map<()[s0] -> (s0 - 9223372036854775807)>
#identity_bound = affine_map<(d0) -> (d0)>
#plus_one = affine_map<(d0) -> (d0 + 1)>
#zero_trip_upper = affine_map<() -> (8, 3)>
#multiple_trip_upper = affine_map<() -> (8, 5)>
#subtract_int64_max = affine_map<(d0) -> (d0 - 9223372036854775807)>

func.func private @unknown(memref<5xi32>)

// A symbolic lower bound is allowed when affine analysis can still prove a
// positive constant trip count. Preload indices are formed from the original
// lower-bound map and operands.

// CHECK-LABEL: func.func @dynamic_lower_bound(
// CHECK-SAME: %[[SRC:.*]]: memref<?xi32>, %[[LB:.*]]: index)
// CHECK: %[[PREV:.*]] = affine.load %[[SRC]][symbol(%[[LB]]) - 1] : memref<?xi32>
// CHECK-NEXT: %[[CURRENT:.*]] = affine.load %[[SRC]][symbol(%[[LB]])] : memref<?xi32>
// CHECK: %{{.*}}:2 = affine.for %[[I:.*]] = %[[LB]] to #{{.*}}()[%[[LB]]]
// CHECK-SAME: iter_args(%[[S0:.*]] = %[[PREV]], %[[S1:.*]] = %[[CURRENT]])
// CHECK: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1] : memref<?xi32>
// CHECK: affine.yield %[[S1]], %[[NEXT]] : i32, i32
func.func @dynamic_lower_bound(%src: memref<?xi32>, %lb: index) {
  affine.for %i = %lb to #dynamic_upper()[%lb] {
    %previous = affine.load %src[%i - 1] : memref<?xi32>
    %current = affine.load %src[%i] : memref<?xi32>
    %next = affine.load %src[%i + 1] : memref<?xi32>
    %sum0 = arith.addi %previous, %current : i32
    %sum1 = arith.addi %sum0, %next : i32
  }
  return
}

// A constant trip count can remain hidden behind affine.apply chains. Exact
// expansion must preserve this supported symbolic-bound case while rejecting
// chains whose composed constants overflow.

// CHECK-LABEL: func.func @composed_trip_count
// CHECK-SAME: %[[SRC:.*]]: memref<?xi32>, %[[LB:.*]]: index
// CHECK: %[[UB0:.*]] = affine.apply #{{.*}}(%[[LB]])
// CHECK-NEXT: %[[UB1:.*]] = affine.apply #{{.*}}(%[[UB0]])
// CHECK-NEXT: %[[PRELOAD:.*]] = affine.load %[[SRC]][symbol(%[[LB]])] : memref<?xi32>
// CHECK-NEXT: affine.for %[[I:.*]] = %[[LB]] to %[[UB1]] iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1] : memref<?xi32>
// CHECK: affine.yield %[[NEXT]] : i32
func.func @composed_trip_count(%src: memref<?xi32>, %lb: index) {
  %ub0 = affine.apply #plus_one(%lb)
  %ub1 = affine.apply #plus_one(%ub0)
  affine.for %i = %lb to %ub1 {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
  }
  return
}

// An access not expressible as IV plus a constant is not part of the window.
// It remains in place while the recognized accesses are reused.

// CHECK-LABEL: func.func @unrecognized_access_coexists
// CHECK: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][0] : memref<?xi32>
// CHECK: %{{.*}} = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1] : memref<?xi32>
// CHECK-NEXT: %[[OTHER:.*]] = affine.load %[[SRC]][symbol(%{{.*}})] : memref<?xi32>
// CHECK: %[[SUM:.*]] = arith.addi %[[STATE]], %[[NEXT]] : i32
// CHECK: arith.addi %[[SUM]], %[[OTHER]] : i32
// CHECK: affine.yield %[[NEXT]] : i32
func.func @unrecognized_access_coexists(%src: memref<?xi32>, %j: index) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
    %other = affine.load %src[symbol(%j)] : memref<?xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %other : i32
  }
  return
}

// Fully composing affine.apply chains proves that these accesses are the same
// contiguous window as direct IV-plus-constant accesses.

// CHECK-LABEL: func.func @composed_affine_apply
// CHECK: %[[X0:.*]] = affine.load %[[SRC:.*]][0] : memref<6xi32>
// CHECK-NEXT: %[[X1:.*]] = affine.load %[[SRC]][1] : memref<6xi32>
// CHECK: %{{.*}}:2 = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[S0:.*]] = %[[X0]], %[[S1:.*]] = %[[X1]])
// CHECK: %[[I1:.*]] = affine.apply #{{.*}}(%[[I]])
// CHECK-NEXT: %[[I2:.*]] = affine.apply #{{.*}}(%[[I1]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I2]]] : memref<6xi32>
// CHECK: %[[SUM0:.*]] = arith.addi %[[S0]], %[[S1]] : i32
// CHECK: %[[SUM1:.*]] = arith.addi %[[SUM0]], %[[NEXT]] : i32
// CHECK: affine.yield %[[S1]], %[[NEXT]] : i32, i32
func.func @composed_affine_apply(%src: memref<6xi32>) {
  affine.for %i = 0 to 4 {
    %i1 = affine.apply #plus_one(%i)
    %i2 = affine.apply #plus_one(%i1)
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i1] : memref<6xi32>
    %x2 = affine.load %src[%i2] : memref<6xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
  }
  return
}

// Constant SSA operands in affine indices are folded when proving offsets.
// This spelling is common in imported HLS kernels.

// CHECK-LABEL: func.func @constant_ssa_offsets
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[X0:.*]] = affine.load %[[SRC:.*]][0] : memref<6xi32>
// CHECK-NEXT: %[[X1:.*]] = affine.load %[[SRC]][1] : memref<6xi32>
// CHECK-NEXT: %{{.*}}:2 = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[S0:.*]] = %[[X0]], %[[S1:.*]] = %[[X1]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + %[[C2]]] : memref<6xi32>
// CHECK: affine.yield %[[S1]], %[[NEXT]] : i32, i32
func.func @constant_ssa_offsets(%src: memref<6xi32>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i + %c1] : memref<6xi32>
    %x2 = affine.load %src[%i + %c2] : memref<6xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
  }
  return
}

// Disjoint contiguous windows on the same flattened source are reused
// independently. This is the access shape used by row-wise 2D stencils.

// CHECK-LABEL: func.func @disjoint_windows_same_source
// CHECK: %[[X0:.*]] = affine.load %[[SRC:.*]][0] : memref<14xi32>
// CHECK-NEXT: %[[X1:.*]] = affine.load %[[SRC]][1] : memref<14xi32>
// CHECK-NEXT: %[[Y0:.*]] = affine.load %[[SRC]][8] : memref<14xi32>
// CHECK-NEXT: %[[Y1:.*]] = affine.load %[[SRC]][9] : memref<14xi32>
// CHECK-NEXT: %{{.*}}:4 = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[SX0:.*]] = %[[X0]], %[[SX1:.*]] = %[[X1]], %[[SY0:.*]] = %[[Y0]], %[[SY1:.*]] = %[[Y1]])
// CHECK-NEXT: %[[XNEXT:.*]] = affine.load %[[SRC]][%[[I]] + 2] : memref<14xi32>
// CHECK-NEXT: %[[YNEXT:.*]] = affine.load %[[SRC]][%[[I]] + 10] : memref<14xi32>
// CHECK-NEXT: %[[SUM0:.*]] = arith.addi %[[SX0]], %[[SX1]] : i32
// CHECK-NEXT: %[[SUM1:.*]] = arith.addi %[[SUM0]], %[[XNEXT]] : i32
// CHECK-NEXT: %[[SUM2:.*]] = arith.addi %[[SY0]], %[[SY1]] : i32
// CHECK-NEXT: %[[SUM3:.*]] = arith.addi %[[SUM2]], %[[YNEXT]] : i32
// CHECK-NEXT: %[[SUM4:.*]] = arith.addi %[[SUM1]], %[[SUM3]] : i32
// CHECK-NEXT: affine.yield %[[SX1]], %[[XNEXT]], %[[SY1]], %[[YNEXT]] : i32, i32, i32, i32
func.func @disjoint_windows_same_source(%src: memref<14xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<14xi32>
    %x1 = affine.load %src[%i + 1] : memref<14xi32>
    %x2 = affine.load %src[%i + 2] : memref<14xi32>
    %y0 = affine.load %src[%i + 8] : memref<14xi32>
    %y1 = affine.load %src[%i + 9] : memref<14xi32>
    %y2 = affine.load %src[%i + 10] : memref<14xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    %sum2 = arith.addi %y0, %y1 : i32
    %sum3 = arith.addi %sum2, %y2 : i32
    %sum4 = arith.addi %sum1, %sum3 : i32
  }
  return
}

// Nested loops use the same transformation. The first-window preload remains
// inside the outer loop and immediately precedes the rewritten inner loop.

// CHECK-LABEL: func.func @nested_loop
// CHECK: affine.for %{{.*}} = 0 to 2 {
// CHECK-NEXT: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][0] : memref<5xi32>
// CHECK-NEXT: %{{.*}} = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1] : memref<5xi32>
// CHECK: %[[SUM:.*]] = arith.addi %[[STATE]], %[[NEXT]] : i32
// CHECK: affine.yield %[[NEXT]] : i32
func.func @nested_loop(%src: memref<5xi32>) {
  affine.for %outer = 0 to 2 {
    affine.for %i = 0 to 4 {
      %x0 = affine.load %src[%i] : memref<5xi32>
      %x1 = affine.load %src[%i + 1] : memref<5xi32>
      %sum = arith.addi %x0, %x1 : i32
    }
  }
  return
}

// Independent windows on different sources are both reused. The pass
// restarts its walk after rebuilding a loop, so neither source is skipped.

// CHECK-LABEL: func.func @two_sources
// CHECK: %[[A0:.*]] = affine.load %[[A:.*]][0] : memref<5xi32>
// CHECK-NEXT: %[[B0:.*]] = affine.load %[[B:.*]][0] : memref<5xi32>
// CHECK: %{{.*}}:2 = affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[SA:.*]] = %[[A0]], %[[SB:.*]] = %[[B0]])
// CHECK: %[[ANEXT:.*]] = affine.load %[[A]][%[[I]] + 1] : memref<5xi32>
// CHECK-NEXT: %[[BNEXT:.*]] = affine.load %[[B]][%[[I]] + 1] : memref<5xi32>
// CHECK: affine.yield %[[ANEXT]], %[[BNEXT]] : i32, i32
func.func @two_sources(%a: memref<5xi32>, %b: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %a0 = affine.load %a[%i] : memref<5xi32>
    %a1 = affine.load %a[%i + 1] : memref<5xi32>
    %b0 = affine.load %b[%i] : memref<5xi32>
    %b1 = affine.load %b[%i + 1] : memref<5xi32>
    %sum0 = arith.addi %a0, %a1 : i32
    %sum1 = arith.addi %b0, %b1 : i32
    %sum2 = arith.addi %sum0, %sum1 : i32
  }
  return
}

// The remaining functions exercise conservative rejection. They must retain
// their original result-less affine.for form.

// A single iteration has no cross-iteration reuse to exploit.

// CHECK-LABEL: func.func @single_trip
// CHECK: affine.for %{{.*}} = 3 to 4 {
func.func @single_trip(%src: memref<5xi32>) {
  affine.for %i = 3 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
  }
  return
}

// The minimum result of a multi-result upper bound determines the trip count.
// A non-positive result makes this loop empty, so it must not gain a preload.

// CHECK-LABEL: func.func @multi_upper_zero_trip
// CHECK: affine.for %{{.*}} = 3 to min #{{.*}}() {
func.func @multi_upper_zero_trip(%src: memref<6xi32>) {
  affine.for %i = 3 to min #zero_trip_upper() {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i + 1] : memref<6xi32>
  }
  return
}

// When every upper-bound span exceeds one, the minimum is still reusable.

// CHECK-LABEL: func.func @multi_upper_multiple_trip
// CHECK: %[[MULTI_PRELOAD:.*]] = affine.load %[[MULTI_SOURCE:.*]][3] : memref<7xi32>
// CHECK-NEXT: affine.for %{{.*}} = 3 to min #{{.*}}() iter_args(%{{.*}} = %[[MULTI_PRELOAD]])
func.func @multi_upper_multiple_trip(%src: memref<7xi32>) {
  affine.for %i = 3 to min #multiple_trip_upper() {
    %x0 = affine.load %src[%i] : memref<7xi32>
    %x1 = affine.load %src[%i + 1] : memref<7xi32>
  }
  return
}

// CHECK-LABEL: func.func @zero_trip
// CHECK: affine.for %{{.*}} = 4 to 4 {
func.func @zero_trip(%src: memref<8xi32>) {
  affine.for %i = 4 to 4 {
    %x0 = affine.load %src[%i] : memref<8xi32>
    %x1 = affine.load %src[%i + 1] : memref<8xi32>
  }
  return
}

// Computing the trip count by subtracting these individually valid i64 bounds
// overflows. Mathematically this is still a zero-trip loop and must not gain
// any preload before the loop.

// CHECK-LABEL: func.func @overflowing_zero_trip
// CHECK-NEXT: affine.for %{{.*}} = 9223372036854775806 to -9223372036854775807 {
func.func @overflowing_zero_trip(%src: memref<1xi32>) {
  affine.for %i = 9223372036854775806 to -9223372036854775807 {
    %x0 = affine.load %src[%i] : memref<1xi32>
    %x1 = affine.load %src[%i + 1] : memref<1xi32>
  }
  return
}

// The same overflow can be hidden behind constant SSA operands and
// affine.apply chains.

// CHECK-LABEL: func.func @ssa_overflowing_zero_trip
// CHECK: %[[LB0:.*]] = arith.constant 9223372036854775806 : index
// CHECK-NEXT: %[[UB0:.*]] = arith.constant -9223372036854775807 : index
// CHECK-NEXT: %[[LB:.*]] = affine.apply #{{.*}}(%[[LB0]])
// CHECK-NEXT: %[[UB:.*]] = affine.apply #{{.*}}(%[[UB0]])
// CHECK-NEXT: affine.for %{{.*}} = %[[LB]] to %[[UB]] {
func.func @ssa_overflowing_zero_trip(%src: memref<1xi32>) {
  %lb0 = arith.constant 9223372036854775806 : index
  %ub0 = arith.constant -9223372036854775807 : index
  %lb = affine.apply #identity_bound(%lb0)
  %ub = affine.apply #identity_bound(%ub0)
  affine.for %i = %lb to %ub {
    %x0 = affine.load %src[%i] : memref<1xi32>
    %x1 = affine.load %src[%i + 1] : memref<1xi32>
  }
  return
}

// Composing this affine.apply chain would combine constants outside signed
// int64. The mathematical upper bound is below the lower bound, so the pass
// must not wrap it into a positive trip count or add an observable preload.

// CHECK-LABEL: func.func @composed_bounds_overflow
// CHECK-SAME: (%{{.*}}: memref<?xi8>, %[[LOWER:.*]]: index)
// CHECK: %[[UPPER0:.*]] = affine.apply #{{.*}}(%[[LOWER]])
// CHECK: %[[UPPER1:.*]] = affine.apply #{{.*}}(%[[UPPER0]])
// CHECK-NEXT: affine.for %[[I:.*]] = %[[LOWER]] to %[[UPPER1]] {
// CHECK-NEXT: affine.load %{{.*}}[%[[I]]] : memref<?xi8>
// CHECK-NEXT: affine.load %{{.*}}[%[[I]] + 1] : memref<?xi8>
func.func @composed_bounds_overflow(%src: memref<?xi8>, %lower: index) {
  %upper0 = affine.apply #subtract_int64_max(%lower)
  %upper1 = affine.apply #subtract_int64_max(%upper0)
  affine.for %i = %lower to %upper1 {
    %x0 = affine.load %src[%i] : memref<?xi8>
    %x1 = affine.load %src[%i + 1] : memref<?xi8>
  }
  return
}

// Cancelling a common symbolic term must not hide the same overflowing bound
// subtraction. At %n = 0 each bound is representable and the loop is empty.

// CHECK-LABEL: func.func @symbolic_overflowing_zero_trip
// CHECK-NEXT: affine.for %{{.*}} = #{{.*}}()[%{{.*}}] to #{{.*}}()[%{{.*}}] {
func.func @symbolic_overflowing_zero_trip(%src: memref<1xi32>, %n: index) {
  affine.for %i = #overflowing_lower()[%n] to #overflowing_upper()[%n] {
    %x0 = affine.load %src[%i] : memref<1xi32>
    %x1 = affine.load %src[%i + 1] : memref<1xi32>
  }
  return
}

// Constant SSA operands can also make access-minus-IV overflow even though
// every original dynamic access is representable. At the two loop iterations,
// these are the valid indices (1, 2) and (2, 3).

// CHECK-LABEL: func.func @overflowing_access_offset
// CHECK: %[[N:.*]] = arith.constant -9223372036854775807 : index
// CHECK-NEXT: affine.for %{{.*}} = -9223372036854775807 to -9223372036854775805 {
func.func @overflowing_access_offset(%src: memref<4xi8>) {
  %n = arith.constant -9223372036854775807 : index
  affine.for %i = -9223372036854775807 to -9223372036854775805 {
    %x0 = affine.load %src[%i - symbol(%n) + 1] : memref<4xi8>
    %x1 = affine.load %src[%i - symbol(%n) + 2] : memref<4xi8>
  }
  return
}

// CHECK-LABEL: func.func @dynamic_trip
// CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
func.func @dynamic_trip(%src: memref<?xi32>, %n: index) {
  affine.for %i = 0 to %n {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
  }
  return
}

// A negative symbolic span is a zero-trip loop. In particular, it must not be
// interpreted as a large unsigned constant trip count.

// CHECK-LABEL: func.func @negative_symbolic_trip
// CHECK: affine.for %{{.*}} = %{{.*}} to #{{.*}}()[%{{.*}}] {
func.func @negative_symbolic_trip(%src: memref<?xi32>, %n: index) {
  affine.for %i = %n to #decreasing_upper()[%n] {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: func.func @non_unit_step
// CHECK: affine.for %{{.*}} = 0 to 8 step 2 {
func.func @non_unit_step(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 step 2 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
  }
  return
}

// CHECK-LABEL: func.func @duplicate_offset
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @duplicate_offset(%src: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x0_again = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
  }
  return
}

// CHECK-LABEL: func.func @non_contiguous_offsets
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @non_contiguous_offsets(%src: memref<6xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x2 = affine.load %src[%i + 2] : memref<6xi32>
  }
  return
}

// Bare function arguments may alias. Moving source reads ahead of this store
// would therefore be unsafe.

// CHECK-LABEL: func.func @may_alias_write
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @may_alias_write(%src: memref<5xi32>, %dst: memref<4xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.store %x0, %dst[%i] : memref<4xi32>
  }
  return
}

// Calls with unknown effects may modify the source and must block reuse.

// CHECK-LABEL: func.func @unknown_effect
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @unknown_effect(%src: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    func.call @unknown(%src) : (memref<5xi32>) -> ()
  }
  return
}

// CHECK-LABEL: func.func @rank_two
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @rank_two(%src: memref<4x5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i, 0] : memref<4x5xi32>
    %x1 = affine.load %src[%i, 1] : memref<4x5xi32>
  }
  return
}

// A source created inside the loop cannot be preloaded before the loop.

// CHECK-LABEL: func.func @source_defined_inside
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NEXT: %[[LOCAL:.*]] = memref.alloca() : memref<5xi32>
// CHECK-NEXT: affine.load %[[LOCAL]][%{{.*}}] : memref<5xi32>
// CHECK-NEXT: affine.load %[[LOCAL]][%{{.*}} + 1] : memref<5xi32>
func.func @source_defined_inside() {
  affine.for %i = 0 to 4 {
    %local = memref.alloca() : memref<5xi32>
    %x0 = affine.load %local[%i] : memref<5xi32>
    %x1 = affine.load %local[%i + 1] : memref<5xi32>
  }
  return
}

// CHECK-LABEL: func.func @writes_source
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @writes_source(%src: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.store %x0, %src[%i + 1] : memref<5xi32>
  }
  return
}

// A write through a cast of the source aliases the preloaded values and must
// also block reuse.

// CHECK-LABEL: func.func @cast_alias_write
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
func.func @cast_alias_write(%src: memref<5xi32>) {
  %alias = memref.cast %src : memref<5xi32> to memref<?xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.store %x0, %alias[%i + 1] : memref<?xi32>
  }
  return
}
