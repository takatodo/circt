// RUN: circt-opt --convert-affine-to-loopschedule --verify-diagnostics %s

func.func @conditional_write(%arg0: memref<2xi32>, %condition: i1) {
  %out = memref.alloca() : memref<2xi32>
  // expected-error@+1 {{unresolved affine memory dependence}}
  affine.for %i = 0 to 2 {
    %value = affine.load %arg0[%i] : memref<2xi32>
    %result = scf.if %condition -> i32 {
      affine.store %value, %arg0[%i] : memref<2xi32>
      scf.yield %value : i32
    } else {
      scf.yield %value : i32
    }
    affine.store %result, %out[%i] : memref<2xi32>
  }
  return
}
