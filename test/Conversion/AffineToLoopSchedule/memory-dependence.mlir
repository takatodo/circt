// RUN: circt-opt --convert-affine-to-loopschedule %s | FileCheck %s

// The loop-carried memory recurrence has latency 5:
// load (1) -> multiply (3) -> add (0) -> store (1).
// Without it, the resource constraints alone permit II 2. The distinct,
// read-only load chain populates intermediate stages without forwarding-only
// stages.

// CHECK-LABEL: func.func @loop_carried_memory_dependence
// CHECK: loopschedule.pipeline II = 5
func.func @loop_carried_memory_dependence(
    %stateArg: memref<16xi32>, %aux0Arg: memref<16xi32>,
    %aux1Arg: memref<?xi32>, %aux2Arg: memref<?xi32>,
    %aux3Arg: memref<?xi32>, %factor: i32) {
  %state, %aux0, %aux1, %aux2, %aux3 = memref.distinct_objects
      %stateArg, %aux0Arg, %aux1Arg, %aux2Arg, %aux3Arg
      : memref<16xi32>, memref<16xi32>, memref<?xi32>,
        memref<?xi32>, memref<?xi32>
  affine.for %i = 1 to 16 {
    %previous = affine.load %state[%i - 1] : memref<16xi32>
    %next = arith.muli %previous, %factor : i32

    %v0 = memref.load %aux0[%i] : memref<16xi32>
    %idx1 = arith.index_cast %v0 : i32 to index
    %v1 = memref.load %aux1[%idx1] : memref<?xi32>
    %idx2 = arith.index_cast %v1 : i32 to index
    %v2 = memref.load %aux2[%idx2] : memref<?xi32>
    %idx3 = arith.index_cast %v2 : i32 to index
    %v3 = memref.load %aux3[%idx3] : memref<?xi32>

    %combined = arith.addi %next, %v3 : i32
    affine.store %combined, %state[%i] : memref<16xi32>
  }
  return
}
