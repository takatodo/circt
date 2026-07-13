// RUN: circt-opt %s -lower-loopschedule-to-calyx -canonicalize -split-input-file | FileCheck %s

// This will introduce duplicate groups; these should be subsequently removed.

// CHECK:      calyx.while %std_lt_0.out with @bb0_0 {
// CHECK-NEXT:  calyx.par {
// CHECK-NEXT:   calyx.enable @bb0_1
// CHECK-NEXT:  }
// CHECK-NEXT: }
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 20 iter_args(%counter = %const) : (index) -> () {
      %latch = arith.cmpi ult, %counter, %const : index
      loopschedule.register %latch : i1
    } do {
      %S0 = loopschedule.pipeline.stage start = 0 {
        %op = arith.addi %counter, %const : index
        loopschedule.register %op : index
      } : index
      %S1 = loopschedule.pipeline.stage start = 1 {
        loopschedule.register %S0: index
      } : index
      loopschedule.terminator iter_args(%S0), results() : (index) -> ()
    }
    return
  }
}

// -----

// Stage pipeline registers passed directly to the next stage 
// should also be updated when used in computations.

// CHECK:      calyx.group @bb0_2 {
// CHECK-NEXT:   calyx.assign %std_add_1.left = %while_0_arg0_reg.out : i32
// CHECK-NEXT:   calyx.assign %std_add_1.right = %c1_i32 : i32
// CHECK-NEXT:   calyx.assign %stage_1_register_0_reg.in = %std_add_1.out : i32
// CHECK-NEXT:   calyx.assign %stage_1_register_0_reg.write_en = %true : i1
// CHECK-NEXT:   calyx.group_done %stage_1_register_0_reg.done : i1
// CHECK-NEXT: }
module {
  func.func @foo() attributes {} {
    %const = arith.constant 1 : index
    loopschedule.pipeline II = 1 trip_count = 20 iter_args(%counter = %const) : (index) -> () {
      %latch = arith.cmpi ult, %counter, %const : index
      loopschedule.register %latch : i1
    } do {
      %S0 = loopschedule.pipeline.stage start = 0 {
        %op = arith.addi %counter, %const : index
        loopschedule.register %op : index
      } : index
      %S1 = loopschedule.pipeline.stage start = 1 {
        %math = arith.addi %S0, %const : index
        loopschedule.register %math : index
      } : index
      loopschedule.terminator iter_args(%S0), results() : (index) -> ()
    }
    return
  }
}

// -----

// A stage may register an iteration argument unchanged. Its result must still
// be replaced with the iteration register output before lowering return-value
// assignments.

// CHECK:      calyx.component @identity_loop
// CHECK:      calyx.group @ret_assign_0 {
// CHECK-NEXT:   calyx.assign %ret_arg0_reg.in = %while_0_arg1_reg.out : i32

module {
  func.func @identity_loop() -> i32 {
    %zero = arith.constant 0 : i32
    %four = arith.constant 4 : i32
    %one = arith.constant 1 : i32
    %init = arith.constant 42 : i32
    %result = loopschedule.pipeline II = 1 trip_count = 4
        iter_args(%counter = %zero, %value = %init) : (i32, i32) -> i32 {
      %condition = arith.cmpi ult, %counter, %four : i32
      loopschedule.register %condition : i1
    } do {
      %stage:2 = loopschedule.pipeline.stage start = 0 {
        %next = arith.addi %counter, %one : i32
        loopschedule.register %value, %next : i32, i32
      } : i32, i32
      loopschedule.terminator iter_args(%stage#1, %stage#0),
          results(%stage#0) : (i32, i32) -> i32
    }
    return %result : i32
  }
}
