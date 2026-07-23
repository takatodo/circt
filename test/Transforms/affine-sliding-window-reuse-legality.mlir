// RUN: circt-opt --affine-sliding-window-reuse %s > %t.once
// RUN: circt-opt --affine-sliding-window-reuse %t.once > %t.twice
// RUN: diff %t.once %t.twice
// RUN: FileCheck %s < %t.once

#positive_upper = affine_map<() -> (4, 6)>
#first_empty_upper = affine_map<() -> (-1, 4)>
#nonlinear_lower = affine_map<()[s0, s1] -> (s0 * (s1 + 1))>
#nonlinear_upper = affine_map<()[s0, s1] -> (s0 * (s1 + 1) + 4)>
#dynamic_plus_nine = affine_map<()[s0] -> (s0 + 9)>
#identity_symbol = affine_map<()[s0] -> (s0)>
#identity_dim = affine_map<(d0) -> (d0)>
#negative_double_dim = affine_map<(d0) -> (-2 * d0)>
#add_pair = affine_map<(d0, d1) -> (d0 + d1)>
#add_pair_plus_four = affine_map<(d0, d1) -> (d0 + d1 + 4)>
#negative_double_plus_four = affine_map<()[s0] -> (-2 * s0 + 4)>
#mixed_lower = affine_map<(d0)[s0] -> (d0 + s0)>
#mixed_upper = affine_map<(d0)[s0] -> (d0 + s0 + 4)>

memref.global "private" @g : memref<5xi32> = dense<[1, 2, 3, 4, 5]>

func.func private @clobber_g() {
  %g = memref.get_global @g : memref<5xi32>
  %zero = arith.constant 0 : i32
  affine.store %zero, %g[1] : memref<5xi32>
  return
}

// Loading through a view and writing through its base is the reverse of the
// usual derived-alias pattern. Hoisting the view loads would still be unsafe.

// CHECK-LABEL: func.func @write_base_of_source_view
// CHECK: memref.subview
// CHECK-NEXT: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
func.func @write_base_of_source_view(%base: memref<5xi32>) {
  %view = memref.subview %base[0] [5] [1]
      : memref<5xi32> to memref<5xi32, strided<[1]>>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %view[%i] : memref<5xi32, strided<[1]>>
    %x1 = affine.load %view[%i + 1] : memref<5xi32, strided<[1]>>
    affine.store %x0, %base[%i + 1] : memref<5xi32>
  }
  return
}

// The source-stability proof is based on generic ModRef information, not an
// affine.store whitelist. A multi-effect copy into the source must block reuse.

// CHECK-LABEL: func.func @copy_writes_source
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
// CHECK: memref.copy
func.func @copy_writes_source(
    %src: memref<5xi32>, %replacement: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    memref.copy %replacement, %src : memref<5xi32> to memref<5xi32>
  }
  return
}

// A region-bearing memory operation can report a write on the parent even
// though its nested operations have no memory effects. Carrying x1 past this
// atomic update would give the next iteration the value from before the RMW.

// CHECK-LABEL: func.func @generic_atomic_rmw_writes_source
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
// CHECK: memref.generic_atomic_rmw
func.func @generic_atomic_rmw_writes_source(
    %src0: memref<5xi32>, %out0: memref<4xi32>) {
  %src, %out = memref.distinct_objects %src0, %out0
      : memref<5xi32>, memref<4xi32>
  %one_index = arith.constant 1 : index
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    %sum = arith.addi %x0, %x1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
    %next_index = arith.addi %i, %one_index : index
    %updated =
        memref.generic_atomic_rmw %src[%next_index] : memref<5xi32> {
      ^bb0(%old : i32):
        %one = arith.constant 1 : i32
        %next = arith.addi %old, %one : i32
        memref.atomic_yield %next : i32
    }
  }
  return
}

// A runtime stride may equal one for some invocations, but it does not prove
// cross-iteration overlap for every invocation.

// CHECK-LABEL: func.func @variable_induction_scale
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
func.func @variable_induction_scale(%src: memref<?xi32>, %stride: index) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i * symbol(%stride)] : memref<?xi32>
    %x1 = affine.load %src[%i * symbol(%stride) + 1] : memref<?xi32>
  }
  return
}

// A symbol operand follows all dimension operands in an affine map. Treating
// the symbol as dimension zero would change -iv + 2*base into increasing iv.

// CHECK-LABEL: func.func @access_symbol_is_not_dimension
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
func.func @access_symbol_is_not_dimension(
    %src: memref<6xi32>, %base: index) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[-%i + 2 * symbol(%base)] : memref<6xi32>
    %x1 = affine.load %src[-%i + 2 * symbol(%base) + 1] : memref<6xi32>
  }
  return
}

// Instability is source-specific. A modified source must be left alone without
// hiding a legal window on a proven-distinct source in the same loop.

// CHECK-LABEL: func.func @modified_source_and_distinct_safe_source
// CHECK: %[[DISTINCT:.*]]:3 = memref.distinct_objects
// CHECK: %[[PRELOAD:.*]] = affine.load %[[DISTINCT]]#1[0]
// CHECK: affine.for %[[I:.*]] = 0 to 4
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK: %[[A0:.*]] = affine.load %[[DISTINCT]]#0[%[[I]]]
// CHECK: %[[A1:.*]] = affine.load %[[DISTINCT]]#0[%[[I]] + 1]
// CHECK: %[[BNEXT:.*]] = affine.load %[[DISTINCT]]#1[%[[I]] + 1]
// CHECK: %[[AS:.*]] = arith.addi %[[A0]], %[[A1]]
// CHECK: %[[BS:.*]] = arith.addi %[[STATE]], %[[BNEXT]]
// CHECK: affine.store %[[AS]], %[[DISTINCT]]#0[%[[I]]]
// CHECK: affine.store %[[BS]], %[[DISTINCT]]#2[%[[I]]]
// CHECK: affine.yield %[[BNEXT]] : i32
func.func @modified_source_and_distinct_safe_source(
    %a0: memref<5xi32>, %b0: memref<5xi32>, %out0: memref<4xi32>) {
  %a, %b, %out = memref.distinct_objects %a0, %b0, %out0
      : memref<5xi32>, memref<5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %a0v = affine.load %a[%i] : memref<5xi32>
    %a1v = affine.load %a[%i + 1] : memref<5xi32>
    %b0v = affine.load %b[%i] : memref<5xi32>
    %b1v = affine.load %b[%i + 1] : memref<5xi32>
    %as = arith.addi %a0v, %a1v : i32
    %bs = arith.addi %b0v, %b1v : i32
    affine.store %as, %a[%i] : memref<5xi32>
    affine.store %bs, %out[%i] : memref<4xi32>
  }
  return
}

// Every result of a multiple-result upper bound participates in the exact
// trip-count proof. The minimum positive span is the actual trip count.

// CHECK-LABEL: func.func @positive_multiple_upper_bounds
// CHECK: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][0]
// CHECK: affine.for %[[I:.*]] = 0 to min #{{.*}}()
// CHECK-SAME: iter_args(%[[STATE:.*]] = %[[PRELOAD]])
// CHECK-NEXT: %[[NEXT:.*]] = affine.load %[[SRC]][%[[I]] + 1]
// CHECK-NEXT: affine.yield %[[NEXT]] : i32
func.func @positive_multiple_upper_bounds(%src: memref<5xi32>) {
  affine.for %i = 0 to min #positive_upper() {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
  }
  return
}

// A non-positive result anywhere in a multiple-result upper bound makes the
// loop empty. Ignoring the first result would hoist an out-of-bounds preload.

// CHECK-LABEL: func.func @first_upper_result_makes_loop_empty
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = 0 to min #{{.*}}() {
func.func @first_upper_result_makes_loop_empty(%src: memref<0xi32>) {
  affine.for %i = 0 to min #first_empty_upper() {
    %x0 = affine.load %src[%i] : memref<0xi32>
    %x1 = affine.load %src[%i + 1] : memref<0xi32>
  }
  return
}

// A product of runtime symbols is not linear. Although the two bounds look
// structurally similar, different operands make the span runtime-dependent
// and possibly zero; for example, n=1, m=4, p=0 gives lower=upper=5.

// CHECK-LABEL: func.func @nonlinear_symbolic_trip
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = #{{.*}}()[%{{.*}}, %{{.*}}] to #{{.*}}()[%{{.*}}, %{{.*}}] {
func.func @nonlinear_symbolic_trip(
    %src: memref<?xi32>, %n: index, %m: index, %p: index) {
  affine.for %i = #nonlinear_lower()[%n, %m]
      to #nonlinear_upper()[%n, %p] {
    %x0 = affine.load %src[%i] : memref<?xi32>
    %x1 = affine.load %src[%i + 1] : memref<?xi32>
  }
  return
}

// Dimension and symbol positions are separate namespaces. The span here is
// hi-lo+4, not a constant four. With base=0, lo=5, hi=1 the loop is empty and
// an unconditional preload at index 5 would be out of bounds.

// CHECK-LABEL: func.func @mixed_dim_symbol_bounds
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = #{{.*}}(%{{.*}})[%{{.*}}] to #{{.*}}(%{{.*}})[%{{.*}}] {
func.func @mixed_dim_symbol_bounds(
    %src: memref<5xi32>, %base: index, %lo: index, %hi: index) {
  affine.for %i = #mixed_lower(%base)[%lo]
      to #mixed_upper(%base)[%hi] {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
  }
  return
}

// Both bounds must have the same coefficient set. Comparing only the lower
// bound's empty set would mistake the constant parts' difference for a trip
// count of four. With n=-4, this loop is actually empty and index 5 is OOB.

// CHECK-LABEL: func.func @missing_upper_coefficient
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = 5 to #{{.*}}()[%{{.*}}] {
func.func @missing_upper_coefficient(
    %src: memref<5xi32>, %n: index) {
  affine.for %i = 5 to #dynamic_plus_nine()[%n] {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
  }
  return
}

// Adding two signed coefficients needs one more bit. If 1 + 1 wraps to -2,
// these opposite-coefficient bounds appear equal with a span of four. For
// n=1 the real loop is empty and preloading index 2 would be OOB.

// CHECK-LABEL: func.func @coefficient_addition_needs_growth
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = %{{.*}} to %{{.*}} {
func.func @coefficient_addition_needs_growth(
    %src: memref<2xi32>, %n: index) {
  %lhs0 = affine.apply #identity_symbol()[%n]
  %lhs1 = affine.apply #identity_symbol()[%n]
  %lower = affine.apply #add_pair(%lhs0, %lhs1)
  %upper = affine.apply #negative_double_plus_four()[%n]
  affine.for %i = %lower to %upper {
    %x0 = affine.load %src[%i] : memref<2xi32>
    %x1 = affine.load %src[%i + 1] : memref<2xi32>
  }
  return
}

// A negative invariant coefficient remains a dependency. Erasing it while
// adding an affine.apply DAG would turn a runtime base into constant zero.

// CHECK-LABEL: func.func @negative_invariant_created_by_dag_sum
// CHECK: affine.for
// CHECK-NOT: iter_args
func.func @negative_invariant_created_by_dag_sum(
    %src0: memref<6xi32>, %dst0: memref<4xi32>, %base: index) {
  %src, %dst = memref.distinct_objects %src0, %dst0
      : memref<6xi32>, memref<4xi32>
  %positive = affine.apply #identity_dim(%base)
  %negative = affine.apply #negative_double_dim(%base)
  %combined = affine.apply #add_pair(%positive, %negative)
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i + symbol(%combined)] : memref<6xi32>
    %x1 = affine.load %src[%i + symbol(%combined) + 1] : memref<6xi32>
    %value = arith.addi %x0, %x1 : i32
    affine.store %value, %dst[%i] : memref<4xi32>
  }
  return
}

// Trip-count expansion uses the same algebra over arbitrary-precision
// coefficients. For n=5 this upper bound is -1, so a preload is invalid.

// CHECK-LABEL: func.func @negative_bound_coefficient_created_by_dag_sum
// CHECK-NOT: affine.load
// CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
func.func @negative_bound_coefficient_created_by_dag_sum(
    %src: memref<0xi32>, %n: index) {
  %positive = affine.apply #identity_dim(%n)
  %negative = affine.apply #negative_double_dim(%n)
  %upper = affine.apply #add_pair_plus_four(%positive, %negative)
  affine.for %i = 0 to %upper {
    %x0 = affine.load %src[%i] : memref<0xi32>
    %x1 = affine.load %src[%i + 1] : memref<0xi32>
  }
  return
}

// Calls without memref operands may still modify a global that aliases the
// source. Looking only at call operands would miss this clobber.

// CHECK-LABEL: func.func @operandless_call_modifies_global_source
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
// CHECK: func.call @clobber_g
func.func @operandless_call_modifies_global_source(
    %out0: memref<4xi32>) {
  %global = memref.get_global @g : memref<5xi32>
  %src, %out = memref.distinct_objects %global, %out0
      : memref<5xi32>, memref<4xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    %sum = arith.addi %x0, %x1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
    func.call @clobber_g() : () -> ()
  }
  return
}

// Alias provenance must follow every scf.if yield. In the false branch the
// result is the sliding source itself, so the following store blocks reuse.

// CHECK-LABEL: func.func @if_result_may_alias_source
// CHECK: affine.for %{{.*}} = 0 to 4 {
// CHECK-NOT: iter_args
// CHECK: scf.if
func.func @if_result_may_alias_source(
    %src0: memref<5xi32>, %other0: memref<5xi32>,
    %out0: memref<4xi32>, %chooseOther: i1) {
  %src, %other, %out =
      memref.distinct_objects %src0, %other0, %out0
      : memref<5xi32>, memref<5xi32>, memref<4xi32>
  %zero = arith.constant 0 : i32
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    %sum = arith.addi %x0, %x1 : i32
    affine.store %sum, %out[%i] : memref<4xi32>
    %alias = scf.if %chooseOther -> (memref<5xi32>) {
      scf.yield %other : memref<5xi32>
    } else {
      scf.yield %src : memref<5xi32>
    }
    affine.store %zero, %alias[%i + 1] : memref<5xi32>
  }
  return
}
