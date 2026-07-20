// RUN: circt-opt --convert-affine-to-loopschedule --verify-diagnostics %s

func.func @unknown_conditional_dependence(
    %arg0: memref<3xi32>, %condition: i1) {
  // expected-error@+1 {{failed to construct scheduling problem due to an unknown memory dependence}}
  affine.for %i = 1 to 3 {
    %previous = affine.load %arg0[%i - 1] : memref<3xi32>
    scf.if %condition {
      affine.store %previous, %arg0[%i] : memref<3xi32>
    }
  }
  return
}
