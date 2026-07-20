// RUN: circt-opt %s -lower-loopschedule-to-calyx -verify-diagnostics

module {
  func.func @unsupported_initiation_interval() {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    // expected-error @+1 {{only pipelines with II = 1 are supported}}
    loopschedule.pipeline II = 2 trip_count = 1
        iter_args(%counter = %zero) : (i32) -> () {
      %condition = arith.cmpi ult, %counter, %one : i32
      loopschedule.register %condition : i1
    } do {
      %stage = loopschedule.pipeline.stage start = 0 {
        %next = arith.addi %counter, %one : i32
        loopschedule.register %next : i32
      } : i32
      loopschedule.terminator iter_args(%stage), results() : (i32) -> ()
    }
    return
  }
}
