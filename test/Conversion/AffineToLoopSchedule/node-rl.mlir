// RUN: circt-opt -convert-affine-to-loopschedule="scheduler=node-rl" %s | FileCheck %s
// RUN: circt-opt -convert-affine-to-loopschedule="scheduler=node-rl node-rl-mcts-trees=1 node-rl-mcts-simulations=1 node-rl-mcts-tree-width=2 node-rl-mcts-rollout-width=2 node-rl-mcts-time-limit=1" %s | FileCheck %s -check-prefix=MCTS
// RUN: %if or-tools %{ circt-opt -convert-affine-to-loopschedule="scheduler=node-rl node-rl-local-search-nodes=8 node-rl-local-search-time-limit=1" %s | FileCheck %s -check-prefix=LNS %}
// RUN: %if or-tools %{ circt-opt -convert-affine-to-loopschedule="scheduler=cpsat" %s | FileCheck %s -check-prefix=CPSAT %}

// The loop-carried accumulator recurrence has a three-cycle multiply. The
// Affine lowering selects Modulo NodeRL and carries its II into LoopSchedule.
// CHECK-LABEL: func @dot_mul_accumulate
// CHECK: loopschedule.pipeline II = 3
// MCTS-LABEL: func @dot_mul_accumulate
// MCTS: loopschedule.pipeline II = 3
// LNS-LABEL: func @dot_mul_accumulate
// LNS: loopschedule.pipeline II = 3
// CPSAT-LABEL: func @dot_mul_accumulate
// CPSAT: loopschedule.pipeline II = 3
func.func @dot_mul_accumulate(%arg0: memref<64xi32>, %arg1: memref<64xi32>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %0 = affine.for %arg2 = 0 to 64 iter_args(%arg3 = %c0_i32) -> (i32) {
    %1 = affine.load %arg0[%arg2] : memref<64xi32>
    %2 = affine.load %arg1[%arg2] : memref<64xi32>
    %3 = arith.muli %1, %2 : i32
    %4 = arith.muli %arg3, %3 : i32
    affine.yield %4 : i32
  }
  return %0 : i32
}

// A memory-carried recurrence must survive affine.load/store lowering.  The
// cycle contains a one-cycle load, a three-cycle multiply, and a one-cycle
// store, which gives a recurrence lower bound of II=5.
// CHECK-LABEL: func @memory_recurrence
// CHECK: loopschedule.pipeline II = 5
// MCTS-LABEL: func @memory_recurrence
// MCTS: loopschedule.pipeline II = 5
// LNS-LABEL: func @memory_recurrence
// LNS: loopschedule.pipeline II = 5
// CPSAT-LABEL: func @memory_recurrence
// CPSAT: loopschedule.pipeline II = 5
func.func @memory_recurrence(%state: memref<33xi32>) {
  affine.for %i = 0 to 32 {
    %previous = affine.load %state[%i] : memref<33xi32>
    %product = arith.muli %previous, %previous : i32
    affine.store %product, %state[%i + 1] : memref<33xi32>
  }
  return
}
