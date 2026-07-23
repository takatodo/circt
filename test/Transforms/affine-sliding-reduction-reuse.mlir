// RUN: circt-opt --affine-sliding-reduction-reuse %s > %t.once
// RUN: circt-opt --affine-sliding-reduction-reuse %t.once > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once
// RUN: circt-opt --affine-sliding-reduction-reuse --canonicalize %s > %t.canonicalized
// RUN: circt-opt %t.canonicalized > /dev/null

#dynamic_upper = affine_map<()[s0] -> (s0 + 4)>
#near_max_apply = affine_map<(d0) -> (d0 + 9223372036854775807)>
#plus_two_symbol = affine_map<()[s0] -> (s0 + 2)>

func.func private @unknown(memref<8xi32>)

// One-row columns carry the raw values. The original leading loads and fold
// tree become dead.

// CHECK-LABEL: func.func @box3_i32
// CHECK: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[P0:.*]] = affine.load %{{.*}}[%[[LB]]] : memref<16xi32>
// CHECK-NEXT: %[[P1:.*]] = affine.load %{{.*}}[%[[LB]] + 1] : memref<16xi32>
// CHECK-NEXT: %{{.*}}:2 = affine.for %[[I:.*]] = 0 to 8
// CHECK-SAME: iter_args(%[[S0:.*]] = %[[P0]], %[[S1:.*]] = %[[P1]])
// CHECK-NEXT: %[[NEW:.*]] = affine.load %{{.*}}[%[[I]] + 2] : memref<16xi32>
// CHECK-NEXT: %[[C0:.*]] = arith.addi %[[S0]], %[[S1]] : i32
// CHECK-NEXT: %[[C1:.*]] = arith.addi %[[C0]], %[[NEW]] : i32
// CHECK-NEXT: affine.store %[[C1]]
// CHECK-NEXT: affine.yield %[[S1]], %[[NEW]] : i32, i32
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

// A two-row, three-column max window carries two column maxima. The outer
// induction variable remains part of each normalized row key.

// CHECK-LABEL: func.func @max2x3_i32
// CHECK: affine.for %[[Y:.*]] = 0 to 8 {
// CHECK-NEXT: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[P00:.*]] = affine.load %{{.*}}[%[[Y]], %[[LB]]]
// CHECK-NEXT: %[[P01:.*]] = affine.load %{{.*}}[%[[Y]] + 1, %[[LB]]]
// CHECK-NEXT: %[[P0:.*]] = arith.maxsi %[[P00]], %[[P01]] : i32
// CHECK-NEXT: %[[P10:.*]] = affine.load %{{.*}}[%[[Y]], %[[LB]] + 1]
// CHECK-NEXT: %[[P11:.*]] = affine.load %{{.*}}[%[[Y]] + 1, %[[LB]] + 1]
// CHECK-NEXT: %[[P1:.*]] = arith.maxsi %[[P10]], %[[P11]] : i32
// CHECK-NEXT: %{{.*}}:2 = affine.for %[[X:.*]] = 0 to 8
// CHECK-SAME: iter_args(%[[C0:.*]] = %[[P0]], %[[C1:.*]] = %[[P1]])
// CHECK-NEXT: %[[N0:.*]] = affine.load %{{.*}}[%[[Y]], %[[X]] + 2]
// CHECK-NEXT: %[[N1:.*]] = affine.load %{{.*}}[%[[Y]] + 1, %[[X]] + 2]
// CHECK-NEXT: %[[NEW:.*]] = arith.maxsi %[[N0]], %[[N1]] : i32
// CHECK-NEXT: %[[M0:.*]] = arith.maxsi %[[C0]], %[[C1]] : i32
// CHECK-NEXT: %[[M1:.*]] = arith.maxsi %[[M0]], %[[NEW]] : i32
// CHECK: affine.yield %[[C1]], %[[NEW]] : i32, i32
func.func @max2x3_i32(%src0: memref<10x10xi32>,
                      %dst0: memref<8x8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10x10xi32>, memref<8x8xi32>
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 8 {
      %x00 = affine.load %src[%y, %x] : memref<10x10xi32>
      %x01 = affine.load %src[%y + 1, %x] : memref<10x10xi32>
      %x10 = affine.load %src[%y, %x + 1] : memref<10x10xi32>
      %x11 = affine.load %src[%y + 1, %x + 1] : memref<10x10xi32>
      %x20 = affine.load %src[%y, %x + 2] : memref<10x10xi32>
      %x21 = affine.load %src[%y + 1, %x + 2] : memref<10x10xi32>
      %m0 = arith.maxsi %x00, %x01 : i32
      %m1 = arith.maxsi %m0, %x10 : i32
      %m2 = arith.maxsi %m1, %x11 : i32
      %m3 = arith.maxsi %m2, %x20 : i32
      %m4 = arith.maxsi %m3, %x21 : i32
      affine.store %m4, %dst[%y, %x] : memref<8x8xi32>
    }
  }
  return
}

// Integer multiplication is a reduction kind; this is distinct from a
// multiply used to map each individual leaf.

// CHECK-LABEL: func.func @mul3_i32
// CHECK: iter_args
// CHECK: arith.muli
func.func @mul3_i32(%src0: memref<16xi32>, %dst0: memref<8xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<16xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<16xi32>
    %x1 = affine.load %src[%i + 1] : memref<16xi32>
    %x2 = affine.load %src[%i + 2] : memref<16xi32>
    %product0 = arith.muli %x0, %x1 : i32
    %product1 = arith.muli %product0, %x2 : i32
    affine.store %product1, %dst[%i] : memref<8xi32>
  }
  return
}

// A loop-invariant symbolic base is part of the access key and is reproduced
// in the prologue.

// CHECK-LABEL: func.func @symbolic_base
// CHECK: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: affine.load %{{.*}}[%[[LB]] + symbol(%{{.*}})]
// CHECK-NEXT: affine.load %{{.*}}[%[[LB]] + symbol(%{{.*}}) + 1]
// CHECK: iter_args
func.func @symbolic_base(%src0: memref<?xi32>, %dst0: memref<4xi32>,
                         %base: index) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<?xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i + symbol(%base)] : memref<?xi32>
    %x1 = affine.load %src[%i + symbol(%base) + 1] : memref<?xi32>
    %x2 = affine.load %src[%i + symbol(%base) + 2] : memref<?xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %dst[%i] : memref<4xi32>
  }
  return
}

// Cloning a leading load at the first iteration composes its access with the
// lower bound. A constant hidden behind an affine.apply operand must not
// overflow when the composed access is canonicalized.

// CHECK-LABEL: func.func @overflowing_preload_operand_composition
// CHECK: %[[LB:.*]] = affine.apply
// CHECK: %[[RESULT:.*]] = affine.for %[[I:.*]] = %[[LB]]
// CHECK-SAME: iter_args(%[[ACC:.*]] = %{{.*}}) -> (i8)
// CHECK-NEXT: %[[X1:.*]] = affine.load %{{.*}}[%[[I]] + 1]
// CHECK-NEXT: %[[X2:.*]] = affine.load %{{.*}}[%[[I]] + 2]
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[X1]], %[[X2]]
func.func @overflowing_preload_operand_composition(
    %src: memref<?xi8>, %n: index) -> i8 {
  %zero = arith.constant 0 : i8
  %lb = affine.apply #near_max_apply(%n)
  %result = affine.for %i = %lb to #plus_two_symbol()[%lb]
      iter_args(%acc = %zero) -> i8 {
    %x1 = affine.load %src[%i + 1] : memref<?xi8>
    %x2 = affine.load %src[%i + 2] : memref<?xi8>
    %sum = arith.addi %x1, %x2 : i8
    %next = arith.addi %acc, %sum : i8
    affine.yield %next : i8
  }
  return %result : i8
}

// A symbolic lower bound is materialized before cloning first-iteration
// loads into the prologue.

// CHECK-LABEL: func.func @dynamic_lower_bound
// CHECK-SAME: %[[SRC0:.*]]: memref<?xi32>, %[[DST0:.*]]: memref<4xi32>, %[[LB:.*]]: index
// CHECK: %[[LBV:.*]] = affine.apply #{{.*}}()[%[[LB]]]
// CHECK-NEXT: %[[P0:.*]] = affine.load %{{.*}}[%[[LBV]]] : memref<?xi32>
// CHECK-NEXT: %[[P1:.*]] = affine.load %{{.*}}[%[[LBV]] + 1] : memref<?xi32>
// CHECK: affine.for %[[I:.*]] = %[[LB]] to #{{.*}}()[%[[LB]]]
// CHECK-SAME: iter_args(%{{.*}} = %[[P0]], %{{.*}} = %[[P1]])
func.func @dynamic_lower_bound(%src0: memref<?xi32>,
                               %dst0: memref<4xi32>, %lb: index) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<?xi32>, memref<4xi32>
  affine.for %i = %lb to #dynamic_upper()[%lb] {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
    %x2 = affine.load %src[%i + 2] : memref<?xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %dst[%i - symbol(%lb)] : memref<4xi32>
  }
  return
}

// Existing unrelated loop state remains first, and all uses of the reduction
// result are rewritten.

// CHECK-LABEL: func.func @existing_state_and_multiple_result_uses
// CHECK: %[[P0:.*]] = affine.load
// CHECK-NEXT: %[[P1:.*]] = affine.load
// CHECK: %[[RESULT:.*]]:3 = affine.for
// CHECK-SAME: iter_args(%[[ACC:.*]] = %{{.*}}, %[[S0:.*]] = %[[P0]], %[[S1:.*]] = %[[P1]])
// CHECK-NEXT: %[[NEW:.*]] = affine.load %{{.*}}[%{{.*}} + 2]
// CHECK-NEXT: %[[PARTIAL:.*]] = arith.addi %[[S0]], %[[S1]] : i32
// CHECK-NEXT: %[[SUM:.*]] = arith.addi %[[PARTIAL]], %[[NEW]] : i32
// CHECK-NEXT: affine.store %[[SUM]]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[ACC]], %[[SUM]] : i32
// CHECK-NEXT: affine.yield %[[NEXT]], %[[S1]], %[[NEW]] : i32, i32, i32
// CHECK: } {test.marker = "state"}
// CHECK: return %[[RESULT]]#0 : i32
func.func @existing_state_and_multiple_result_uses(
    %src0: memref<16xi32>, %dst0: memref<8xi32>) -> i32 {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<16xi32>, memref<8xi32>
  %zero = arith.constant 0 : i32
  %result = affine.for %i = 0 to 8
      iter_args(%acc = %zero) -> i32 {
    %x0 = affine.load %src[%i] : memref<16xi32>
    %x1 = affine.load %src[%i + 1] : memref<16xi32>
    %x2 = affine.load %src[%i + 2] : memref<16xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %dst[%i] : memref<8xi32>
    %next = arith.addi %acc, %sum1 : i32
    affine.yield %next : i32
  } {test.marker = "state"}
  return %result : i32
}

// A load-only reduction remains a candidate when its sole user is the same
// operation kind with a loop-carried accumulator operand.

// CHECK-LABEL: func.func @seeded_accumulator
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[LB:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[P00:.*]] = affine.load %[[SRC:.*]][0, %[[LB]]]
// CHECK-NEXT: %[[P01:.*]] = affine.load %[[SRC]][1, %[[LB]]]
// CHECK-NEXT: %[[P0:.*]] = arith.addi %[[P00]], %[[P01]] : i32
// CHECK-NEXT: %[[P10:.*]] = affine.load %[[SRC]][0, %[[LB]] + 1]
// CHECK-NEXT: %[[P11:.*]] = affine.load %[[SRC]][1, %[[LB]] + 1]
// CHECK-NEXT: %[[P1:.*]] = arith.addi %[[P10]], %[[P11]] : i32
// CHECK: %[[RESULT:.*]]:3 = affine.for %[[I:.*]] = 0 to 8
// CHECK-SAME: iter_args(%[[ACC:.*]] = %[[ZERO]], %[[C0:.*]] = %[[P0]], %[[C1:.*]] = %[[P1]])
// CHECK: %[[N0:.*]] = affine.load %[[SRC]][0, %[[I]] + 2]
// CHECK-NEXT: %[[N1:.*]] = affine.load %[[SRC]][1, %[[I]] + 2]
// CHECK-NEXT: %[[NEW:.*]] = arith.addi %[[N0]], %[[N1]] : i32
// CHECK-NEXT: %[[PARTIAL:.*]] = arith.addi %[[C0]], %[[C1]] : i32
// CHECK-NEXT: %[[WINDOW:.*]] = arith.addi %[[PARTIAL]], %[[NEW]] : i32
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[ACC]], %[[WINDOW]] : i32
// CHECK-NEXT: affine.yield %[[NEXT]], %[[C1]], %[[NEW]] : i32, i32, i32
// CHECK: return %[[RESULT]]#0 : i32
func.func @seeded_accumulator(%src: memref<2x10xi32>) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %i = 0 to 8
      iter_args(%acc = %zero) -> i32 {
    %x00 = affine.load %src[0, %i] : memref<2x10xi32>
    %x01 = affine.load %src[1, %i] : memref<2x10xi32>
    %x10 = affine.load %src[0, %i + 1] : memref<2x10xi32>
    %x11 = affine.load %src[1, %i + 1] : memref<2x10xi32>
    %x20 = affine.load %src[0, %i + 2] : memref<2x10xi32>
    %x21 = affine.load %src[1, %i + 2] : memref<2x10xi32>
    %sum0 = arith.addi %x00, %x01 : i32
    %sum1 = arith.addi %sum0, %x10 : i32
    %sum2 = arith.addi %sum1, %x11 : i32
    %sum3 = arith.addi %sum2, %x20 : i32
    %window = arith.addi %sum3, %x21 : i32
    %next = arith.addi %acc, %window : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// A leading load with another user would remain in the loop, so carrying it
// would add state without reducing the load count.

// CHECK-LABEL: func.func @leading_load_external_use
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @leading_load_external_use(
    %src0: memref<10xi32>, %folded0: memref<8xi32>,
    %side0: memref<8xi32>) {
  %src, %folded, %side =
      memref.distinct_objects %src0, %folded0, %side0
      : memref<10xi32>, memref<8xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %x2 = affine.load %src[%i + 2] : memref<10xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    affine.store %sum1, %folded[%i] : memref<8xi32>
    %difference = arith.subi %x0, %x1 : i32
    affine.store %difference, %side[%i] : memref<8xi32>
  }
  return
}

// A dead fold is left for dead-code elimination instead of gaining prologue
// work and loop-carried state.

// CHECK-LABEL: func.func @dead_fold
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @dead_fold(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %x2 = affine.load %src[%i + 2] : memref<10xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
  }
  return
}

// A subtree with both a same-kind parent and another live use is an
// independent candidate. Replacing it must update both users.

// CHECK-LABEL: func.func @shared_live_subtree
// CHECK: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][%{{.*}}]
// CHECK: affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1]
// CHECK-NEXT: %[[X2:.*]] = affine.load %[[SRC]][%[[I]] + 2]
// CHECK-NEXT: %[[WINDOW:.*]] = arith.addi %[[STATE]], %[[NEXT]] : i32
// CHECK-NEXT: %[[OUTER:.*]] = arith.addi %[[WINDOW]], %[[X2]] : i32
// CHECK-NEXT: %[[SIDE:.*]] = arith.subi %[[WINDOW]], %{{.*}} : i32
// CHECK-NEXT: affine.store %[[OUTER]], %{{.*}}[%[[I]], 0]
// CHECK-NEXT: affine.store %[[SIDE]], %{{.*}}[%[[I]], 1]
// CHECK: affine.yield %[[NEXT]] : i32
func.func @shared_live_subtree(%src0: memref<6xi32>,
                               %dst0: memref<4x2xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<6xi32>, memref<4x2xi32>
  %one = arith.constant 1 : i32
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i + 1] : memref<6xi32>
    %x2 = affine.load %src[%i + 2] : memref<6xi32>
    %window = arith.addi %x0, %x1 : i32
    %outer = arith.addi %window, %x2 : i32
    %side = arith.subi %window, %one : i32
    affine.store %outer, %dst[%i, 0] : memref<4x2xi32>
    affine.store %side, %dst[%i, 1] : memref<4x2xi32>
  }
  return
}

// Independent windows in one loop are found across fixpoint iterations.

// CHECK-LABEL: func.func @two_windows
// CHECK: %[[AP:.*]] = affine.load %[[A:.*]][%{{.*}}] : memref<9xi32>
// CHECK: %[[BP:.*]] = affine.load %[[B:.*]][%{{.*}}] : memref<9xi32>
// CHECK: affine.for %[[I:.*]] = 0 to 8
// CHECK-SAME: iter_args(%[[AS:.*]] = %[[AP]], %[[BS:.*]] = %[[BP]])
// CHECK: %[[ANEXT:.*]] = affine.load %[[A]][%[[I]] + 1]
// CHECK: %[[BNEXT:.*]] = affine.load %[[B]][%[[I]] + 1]
// CHECK: affine.yield %[[ANEXT]], %[[BNEXT]] : i32, i32
func.func @two_windows(%a0: memref<9xi32>, %b0: memref<9xi32>,
                       %x0: memref<8xi32>, %y0: memref<8xi32>) {
  %a, %b, %x, %y = memref.distinct_objects %a0, %b0, %x0, %y0
      : memref<9xi32>, memref<9xi32>, memref<8xi32>, memref<8xi32>
  affine.for %i = 0 to 8 {
    %a0v = affine.load %a[%i] : memref<9xi32>
    %a1v = affine.load %a[%i + 1] : memref<9xi32>
    %as = arith.addi %a0v, %a1v : i32
    affine.store %as, %x[%i] : memref<8xi32>
    %b0v = affine.load %b[%i] : memref<9xi32>
    %b1v = affine.load %b[%i + 1] : memref<9xi32>
    %bs = arith.addi %b0v, %b1v : i32
    affine.store %bs, %y[%i] : memref<8xi32>
  }
  return
}

// Balanced trees and negative sliding offsets use the same column identity.
// The prologue substitutes the exact first induction-variable value.

// CHECK-LABEL: func.func @balanced_negative_offsets
// CHECK: %[[LB:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[P0:.*]] = affine.load %[[SRC:.*]][0, %[[LB]] - 1]
// CHECK-NEXT: %[[P1:.*]] = affine.load %[[SRC]][1, %[[LB]] - 1]
// CHECK-NEXT: %[[PREV:.*]] = arith.addi %[[P0]], %[[P1]] : i32
// CHECK-NEXT: affine.for %[[I:.*]] = 1 to 5
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PREV]])
// CHECK-NEXT: %[[N0:.*]] = affine.load %[[SRC]][0, %[[I]]]
// CHECK-NEXT: %[[N1:.*]] = affine.load %[[SRC]][1, %[[I]]]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[N0]], %[[N1]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.addi %[[STATE]], %[[NEXT]] : i32
// CHECK-NEXT: affine.store %[[RESULT]]
// CHECK-NEXT: affine.yield %[[NEXT]] : i32
func.func @balanced_negative_offsets(%src0: memref<2x5xi32>,
                                     %dst0: memref<4xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<2x5xi32>, memref<4xi32>
  affine.for %i = 1 to 5 {
    %x00 = affine.load %src[0, %i - 1] : memref<2x5xi32>
    %x01 = affine.load %src[1, %i - 1] : memref<2x5xi32>
    %x10 = affine.load %src[0, %i] : memref<2x5xi32>
    %x11 = affine.load %src[1, %i] : memref<2x5xi32>
    %previous = arith.addi %x00, %x01 : i32
    %current = arith.addi %x10, %x11 : i32
    %result = arith.addi %previous, %current : i32
    affine.store %result, %dst[%i - 1] : memref<4xi32>
  }
  return
}

// Floating-point folds require reassociation and retain it on new operations.

// CHECK-LABEL: func.func @addf_reassoc
// CHECK: iter_args
// CHECK-COUNT-2: arith.addf %{{.*}}, %{{.*}} fastmath<reassoc> : f32
func.func @addf_reassoc(%src0: memref<10xf32>, %dst0: memref<8xf32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<10xf32>, memref<8xf32>
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xf32>
    %x1 = affine.load %src[%i + 1] : memref<10xf32>
    %x2 = affine.load %src[%i + 2] : memref<10xf32>
    %sum0 = arith.addf %x0, %x1 fastmath<reassoc,nnan> : f32
    %sum1 = arith.addf %sum0, %x2 fastmath<reassoc,ninf> : f32
    affine.store %sum1, %dst[%i] : memref<8xf32>
  }
  return
}

// The remaining cases must retain their result-less affine.for form.

// CHECK-LABEL: func.func @single_trip
// CHECK: affine.for %{{.*}} = 3 to 4 {
// CHECK-NOT: iter_args
func.func @single_trip(%src: memref<6xi32>) {
  affine.for %i = 3 to 4 {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i + 1] : memref<6xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// CHECK-LABEL: func.func @overflowing_zero_trip
// CHECK: affine.for %{{.*}} = 9223372036854775806 to -9223372036854775807 {
// CHECK-NOT: iter_args
func.func @overflowing_zero_trip(%src: memref<1xi32>) {
  affine.for %i = 9223372036854775806 to -9223372036854775807 {
    %x0 = affine.load %src[%i] : memref<1xi32>
    %x1 = affine.load %src[%i + 1] : memref<1xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// CHECK-LABEL: func.func @dynamic_trip
// CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK-NOT: iter_args
func.func @dynamic_trip(%src: memref<?xi32>, %n: index) {
  affine.for %i = 0 to %n {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// CHECK-LABEL: func.func @non_unit_step
// CHECK: affine.for %{{.*}} = 0 to 8 step 2 {
// CHECK-NOT: iter_args
func.func @non_unit_step(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 step 2 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// Overflow flags make integer regrouping poison-sensitive.

// CHECK-LABEL: func.func @addi_nsw
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @addi_nsw(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %x2 = affine.load %src[%i + 2] : memref<10xi32>
    %sum0 = arith.addi %x0, %x1 overflow<nsw> : i32
    %sum1 = arith.addi %sum0, %x2 overflow<nsw> : i32
    %used = arith.cmpi eq, %sum1, %sum1 : i32
  }
  return
}

// Every internal node, not only the root, must permit integer regrouping.

// CHECK-LABEL: func.func @muli_internal_nsw
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @muli_internal_nsw(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %x2 = affine.load %src[%i + 2] : memref<10xi32>
    %product0 = arith.muli %x0, %x1 overflow<nsw> : i32
    %product1 = arith.muli %product0, %x2 : i32
    %used = arith.cmpi eq, %product1, %product1 : i32
  }
  return
}

// Signed and unsigned overflow promises are independent bits. Checking only
// nsw would incorrectly admit this poison-sensitive internal multiplication.

// CHECK-LABEL: func.func @muli_internal_nuw
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @muli_internal_nuw(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %x2 = affine.load %src[%i + 2] : memref<10xi32>
    %product0 = arith.muli %x0, %x1 overflow<nuw> : i32
    %product1 = arith.muli %product0, %x2 : i32
    %used = arith.cmpi eq, %product1, %product1 : i32
  }
  return
}

// Unsupported reduction kinds must not be classified as a supported kind.
// Rebuilding signed minimum as signed maximum would change [5, 2, 7] from
// two to seven.

// CHECK-LABEL: func.func @minsi_is_not_maxsi
// CHECK-NOT: arith.maxsi
// CHECK-COUNT-2: arith.minsi
func.func @minsi_is_not_maxsi(%src0: memref<6xi32>,
                              %dst0: memref<4xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<6xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<6xi32>
    %x1 = affine.load %src[%i + 1] : memref<6xi32>
    %x2 = affine.load %src[%i + 2] : memref<6xi32>
    %min0 = arith.minsi %x0, %x1 : i32
    %min1 = arith.minsi %min0, %x2 : i32
    affine.store %min1, %dst[%i] : memref<4xi32>
  }
  return
}

// Every internal node must have the root's operation kind. Flattening
// a0*b0 + a1*b1 as one add tree would change 2*3 + 4*5 from 26 to 14.

// CHECK-LABEL: func.func @mixed_add_of_products
// CHECK-COUNT-2: arith.muli
// CHECK: arith.addi
func.func @mixed_add_of_products(%src0: memref<2x5xi32>,
                                 %dst0: memref<4xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<2x5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0 = affine.load %src[0, %i] : memref<2x5xi32>
    %b0 = affine.load %src[1, %i] : memref<2x5xi32>
    %a1 = affine.load %src[0, %i + 1] : memref<2x5xi32>
    %b1 = affine.load %src[1, %i + 1] : memref<2x5xi32>
    %p0 = arith.muli %a0, %b0 : i32
    %p1 = arith.muli %a1, %b1 : i32
    %sum = arith.addi %p0, %p1 : i32
    affine.store %sum, %dst[%i] : memref<4xi32>
  }
  return
}

// A legal root does not compensate for a strict internal floating-point node.

// CHECK-LABEL: func.func @addf_internal_strict
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @addf_internal_strict(%src: memref<10xf32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xf32>
    %x1 = affine.load %src[%i + 1] : memref<10xf32>
    %x2 = affine.load %src[%i + 2] : memref<10xf32>
    %sum0 = arith.addf %x0, %x1 : f32
    %sum1 = arith.addf %sum0, %x2 fastmath<reassoc> : f32
    %used = arith.cmpf oeq, %sum1, %sum1 : f32
  }
  return
}

// Explicit rounding on any internal node also prevents reassociation.

// CHECK-LABEL: func.func @addf_internal_rounding
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @addf_internal_rounding(%src: memref<10xf32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xf32>
    %x1 = affine.load %src[%i + 1] : memref<10xf32>
    %x2 = affine.load %src[%i + 2] : memref<10xf32>
    %sum0 = arith.addf %x0, %x1 upward fastmath<reassoc> : f32
    %sum1 = arith.addf %sum0, %x2 fastmath<reassoc> : f32
    %used = arith.cmpf oeq, %sum1, %sum1 : f32
  }
  return
}

// Bare function arguments may alias. Moving source reads ahead of this store
// is therefore unsafe.

// CHECK-LABEL: func.func @may_alias
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @may_alias(%src: memref<10xi32>, %dst: memref<8xi32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%i + 1] : memref<10xi32>
    %sum = arith.addi %x0, %x1 : i32
    affine.store %sum, %dst[%i] : memref<8xi32>
  }
  return
}

// CHECK-LABEL: func.func @unknown_effect
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @unknown_effect(%src: memref<8xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<8xi32>
    %x1 = affine.load %src[%i + 1] : memref<8xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
    func.call @unknown(%src) : (memref<8xi32>) -> ()
  }
  return
}

// Adjacent offsets from different sources are not one sliding window. Carrying
// a value from A as the next iteration's value from B would change the result.

// CHECK-LABEL: func.func @mixed_sources
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @mixed_sources(%a0: memref<5xi32>, %b0: memref<5xi32>) {
  %a, %b = memref.distinct_objects %a0, %b0
      : memref<5xi32>, memref<5xi32>
  affine.for %i = 0 to 4 {
    %x = affine.load %a[%i] : memref<5xi32>
    %y = affine.load %b[%i + 1] : memref<5xi32>
    %sum = arith.addi %x, %y : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// Gaps, ragged columns, and nonlinear subscripts do not describe reusable
// equal-shaped columns.

// Different symbolic bases must not be treated as adjacent columns.

// CHECK-LABEL: func.func @symbolic_base_mismatch
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @symbolic_base_mismatch(%src: memref<?xi32>, %a: index,
                                  %b: index) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i + symbol(%a)] : memref<?xi32>
    %x1 = affine.load %src[%i + symbol(%b) + 1] : memref<?xi32>
    %x2 = affine.load %src[%i + symbol(%a) + 2] : memref<?xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    %used = arith.cmpi eq, %sum1, %sum1 : i32
  }
  return
}

// Consecutive addresses within one iteration do not overlap the next
// iteration when the induction-variable coefficient is two.

// CHECK-LABEL: func.func @scaled_iv
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @scaled_iv(%src: memref<16xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[2 * %i] : memref<16xi32>
    %x1 = affine.load %src[2 * %i + 1] : memref<16xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// Reverse traversal overlaps in the opposite direction. Treating -1 as
// forward unit stride would carry src[5] into iteration one, where x0 is
// src[3].

// CHECK-LABEL: func.func @reverse_unit_induction_is_not_forward_reuse
// CHECK: %[[DISTINCT:.*]]:2 = memref.distinct_objects
// CHECK-NEXT: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
func.func @reverse_unit_induction_is_not_forward_reuse(
    %src0: memref<6xi32>, %dst0: memref<4xi32>) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<6xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[-%i + 4] : memref<6xi32>
    %x1 = affine.load %src[-%i + 5] : memref<6xi32>
    %sum = arith.addi %x0, %x1 : i32
    affine.store %sum, %dst[%i] : memref<4xi32>
  }
  return
}

// CHECK-LABEL: func.func @offset_gap
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @offset_gap(%src: memref<12xi32>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<12xi32>
    %x2 = affine.load %src[%i + 2] : memref<12xi32>
    %sum = arith.addi %x0, %x2 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// CHECK-LABEL: func.func @ragged_columns
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @ragged_columns(%src: memref<10x10xi32>) {
  affine.for %y = 0 to 8 {
    affine.for %x = 0 to 8 {
      %x00 = affine.load %src[%y, %x] : memref<10x10xi32>
      %x01 = affine.load %src[%y + 1, %x] : memref<10x10xi32>
      %x10 = affine.load %src[%y, %x + 1] : memref<10x10xi32>
      %sum0 = arith.addi %x00, %x01 : i32
      %sum1 = arith.addi %sum0, %x10 : i32
      %used = arith.cmpi eq, %sum1, %sum1 : i32
    }
  }
  return
}

// A larger later column is also ragged. Ignoring its unmatched row would drop
// that value from the rebuilt reduction.

// CHECK-LABEL: func.func @larger_later_column
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @larger_later_column(%src: memref<2x5xi32>) {
  affine.for %i = 0 to 4 {
    %x00 = affine.load %src[0, %i] : memref<2x5xi32>
    %x10 = affine.load %src[0, %i + 1] : memref<2x5xi32>
    %x11 = affine.load %src[1, %i + 1] : memref<2x5xi32>
    %later = arith.addi %x10, %x11 : i32
    %sum = arith.addi %x00, %later : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// Equal column sizes are insufficient: corresponding rows must have the same
// normalized access key.

// CHECK-LABEL: func.func @mismatched_row_keys
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @mismatched_row_keys(%src: memref<3x5xi32>) {
  affine.for %i = 0 to 4 {
    %x00 = affine.load %src[0, %i] : memref<3x5xi32>
    %x01 = affine.load %src[1, %i] : memref<3x5xi32>
    %x10 = affine.load %src[0, %i + 1] : memref<3x5xi32>
    %x12 = affine.load %src[2, %i + 1] : memref<3x5xi32>
    %sum0 = arith.addi %x00, %x01 : i32
    %sum1 = arith.addi %sum0, %x10 : i32
    %sum2 = arith.addi %sum1, %x12 : i32
    %used = arith.cmpi eq, %sum2, %sum2 : i32
  }
  return
}

// Row keys form a multiset, not merely a set. Matching both copies of row
// zero in the first column to the single row-zero load in the second would
// duplicate that value and silently discard the row-one load.

// CHECK-LABEL: func.func @mismatched_row_key_multiplicity
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = 0 to 4 {
func.func @mismatched_row_key_multiplicity(%src: memref<2x5xi32>) {
  affine.for %i = 0 to 4 {
    %x00 = affine.load %src[0, %i] : memref<2x5xi32>
    %x00_again = affine.load %src[0, %i] : memref<2x5xi32>
    %x10 = affine.load %src[0, %i + 1] : memref<2x5xi32>
    %x11 = affine.load %src[1, %i + 1] : memref<2x5xi32>
    %sum0 = arith.addi %x00, %x00_again : i32
    %sum1 = arith.addi %sum0, %x10 : i32
    %sum2 = arith.addi %sum1, %x11 : i32
    %used = arith.cmpi eq, %sum2, %sum2 : i32
  }
  return
}

// All loads must slide along the same memref dimension.

// CHECK-LABEL: func.func @sliding_position_mismatch
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @sliding_position_mismatch(%src: memref<5x5xi32>) {
  affine.for %i = 0 to 4 {
    %x = affine.load %src[%i, 0] : memref<5x5xi32>
    %y = affine.load %src[0, %i + 1] : memref<5x5xi32>
    %sum = arith.addi %x, %y : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// A single load that varies in two dimensions has no unique sliding position.

// CHECK-LABEL: func.func @multiple_sliding_subscripts
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @multiple_sliding_subscripts(%src: memref<6x6xi32>) {
  affine.for %i = 0 to 4 {
    %x = affine.load %src[%i, %i] : memref<6x6xi32>
    %y = affine.load %src[%i + 1, %i + 2] : memref<6x6xi32>
    %sum = arith.addi %x, %y : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// CHECK-LABEL: func.func @nonlinear_subscript
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @nonlinear_subscript(%src: memref<8xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i mod 4] : memref<8xi32>
    %x1 = affine.load %src[%i + 1] : memref<8xi32>
    %sum = arith.addi %x0, %x1 : i32
    %used = arith.cmpi eq, %sum, %sum : i32
  }
  return
}

// An affine.apply produced inside the loop cannot be referenced by a cloned
// prologue load even when its expression is otherwise linear.

// CHECK-LABEL: func.func @inloop_apply
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @inloop_apply(%src: memref<10xi32>) {
  affine.for %i = 0 to 8 {
    %j = affine.apply affine_map<(d0) -> (d0 + 1)>(%i)
    %x0 = affine.load %src[%i] : memref<10xi32>
    %x1 = affine.load %src[%j] : memref<10xi32>
    %x2 = affine.load %src[%i + 2] : memref<10xi32>
    %sum0 = arith.addi %x0, %x1 : i32
    %sum1 = arith.addi %sum0, %x2 : i32
    %used = arith.cmpi eq, %sum1, %sum1 : i32
  }
  return
}

// Mapped leaves are not matched.

// CHECK-LABEL: func.func @extension_leaf
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @extension_leaf(%src: memref<10xi8>) {
  affine.for %i = 0 to 8 {
    %x0 = affine.load %src[%i] : memref<10xi8>
    %x1 = affine.load %src[%i + 1] : memref<10xi8>
    %x2 = affine.load %src[%i + 2] : memref<10xi8>
    %w0 = arith.extsi %x0 : i8 to i32
    %w1 = arith.extsi %x1 : i8 to i32
    %w2 = arith.extsi %x2 : i8 to i32
    %sum0 = arith.addi %w0, %w1 : i32
    %sum1 = arith.addi %sum0, %w2 : i32
    %used = arith.cmpi eq, %sum1, %sum1 : i32
  }
  return
}
