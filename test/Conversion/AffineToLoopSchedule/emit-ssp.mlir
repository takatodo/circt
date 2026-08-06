// RUN: circt-opt %s -convert-affine-to-loopschedule="emit-ssp=true multiplier-limit=2 multiplier-ii=3" 2>&1 | FileCheck %s

// CHECK: // ----- BEGIN AFFINE MODULO SSP -----
// CHECK: ssp.instance @affine_modulo_0 of "ModuloProblem" {
// CHECK: operator_type @multicycle [latency<3>]
// CHECK: resource_type @multiplier [limit<2>, ii<3>]
// CHECK: graph {
// CHECK: %op{{[0-9]+}} = operation<@multicycle> @op{{[0-9]+}}({{.*}}) uses[@multiplier]
// CHECK: operation<@comb> @last({{.*}})
// CHECK: // ----- END AFFINE MODULO SSP -----
func.func @memory_recurrence(%state: memref<33xi32>) {
  affine.for %i = 0 to 32 {
    %previous = affine.load %state[%i] : memref<33xi32>
    %product = arith.muli %previous, %previous : i32
    affine.store %product, %state[%i + 1] : memref<33xi32>
  }
  return
}
