// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=3 cpsat-time-limit=2 cpsat-workers=1 cpsat-resource-model=cumulative" | FileCheck %s --check-prefix=STATIC %}
// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=3 cpsat-time-limit=2 cpsat-workers=1 cpsat-resource-model=cumulative" -lower-loopschedule-to-calyx="top-level-function=multiple_multipliers" | FileCheck %s --check-prefix=STATIC-CALYX %}
// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=4 cpsat-time-limit=2 cpsat-workers=1 cpsat-resource-model=cumulative" | FileCheck %s --check-prefix=ROTATING %}
// RUN: %if or-tools %{ circt-opt %s -convert-affine-to-loopschedule="scheduler=cpsat multiplier-limit=2 multiplier-ii=4 cpsat-time-limit=2 cpsat-workers=1 cpsat-resource-model=cumulative" -lower-loopschedule-to-calyx="top-level-function=rotating_multiplier" | FileCheck %s --check-prefix=ROTATING-CALYX %}

// CP-SAT returns start times without physical bindings. Two simultaneous
// operations become distinct period-one selectors and exactly two Calyx cells.
// STATIC-LABEL: func @multiple_multipliers
// STATIC: loopschedule.pipeline II = 3
// STATIC-DAG: resource = "multiplier", selector = [0], selector_period = 1
// STATIC-DAG: resource = "multiplier", selector = [1], selector_period = 1
// STATIC-CALYX-LABEL: calyx.component @multiple_multipliers
// STATIC-CALYX-DAG: calyx.std_mult_pipe @std_mult_pipe_0
// STATIC-CALYX-DAG: calyx.std_mult_pipe @std_mult_pipe_1
// STATIC-CALYX-NOT: calyx.std_mult_pipe @std_mult_pipe_2
// STATIC-CALYX-NOT: rotating_phase
func.func @multiple_multipliers(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %zero = arith.constant 0 : i32
  %results:2 = affine.for %i = 0 to 16
      iter_args(%lhs = %zero, %rhs = %zero) -> (i32, i32) {
    %next_lhs = arith.muli %lhs, %arg0 : i32
    %next_rhs = arith.muli %rhs, %arg1 : i32
    affine.yield %next_lhs, %next_rhs : i32, i32
  }
  return %results#0, %results#1 : i32, i32
}

// A resource hold longer than the pipeline II rotates one logical operation
// between two physical instances in successive iterations.
// ROTATING-LABEL: func @rotating_multiplier
// ROTATING: loopschedule.pipeline II = 3
// ROTATING: resource = "multiplier", selector = [0, 1], selector_period = 2
// ROTATING-CALYX-LABEL: calyx.component @rotating_multiplier
// ROTATING-CALYX: calyx.register @rotating_phase
// ROTATING-CALYX-DAG: calyx.std_mult_pipe @std_mult_pipe_0
// ROTATING-CALYX-DAG: calyx.std_mult_pipe @std_mult_pipe_1
func.func @rotating_multiplier(%arg0: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %result = affine.for %i = 0 to 16 iter_args(%acc = %zero) -> (i32) {
    %next = arith.muli %acc, %arg0 : i32
    affine.yield %next : i32
  }
  return %result : i32
}
