// RUN: circt-opt --affine-sliding-window-reuse %s | FileCheck %s --check-prefix=IR

func.func private @unknown(memref<5xi32>)

// Reads nested under a recursive-effect container also remain legal.

// IR-LABEL: func.func @nested_read_only
// IR: %[[PRELOAD:.*]] = affine.load %[[SRC:.*]][0] : memref<5xi32>
// IR-NEXT: affine.for %[[I:.*]] = 0 to 4 iter_args(%[[STATE:.*]] = %[[PRELOAD]]) -> (i32) {
// IR-NEXT: affine.load %[[SRC]][%[[I]] + 1] : memref<5xi32>
// IR-NEXT: affine.for
// IR-NEXT: affine.load %[[SRC]][0] : memref<5xi32>
func.func @nested_read_only(%src: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.for %j = 0 to 1 {
      %nested = affine.load %src[0] : memref<5xi32>
    }
  }
  return
}

// A nested write to a provably distinct memref is legal. The child store is
// still queried independently by AliasAnalysis.

// IR-LABEL: func.func @nested_distinct_write
// IR: %[[DISTINCT:.*]]:2 = memref.distinct_objects
// IR-NEXT: %[[PRELOAD:.*]] = affine.load %[[DISTINCT]]#0[0] : memref<5xi32>
// IR-NEXT: affine.for %[[I:.*]] = 0 to 4 iter_args(%[[STATE:.*]] = %[[PRELOAD]]) -> (i32) {
// IR-NEXT: affine.load %[[DISTINCT]]#0[%[[I]] + 1] : memref<5xi32>
// IR-NEXT: affine.for
// IR-NEXT: affine.store %[[STATE]], %[[DISTINCT]]#1[0] : memref<1xi32>
func.func @nested_distinct_write(%srcArg: memref<5xi32>,
                                 %dstArg: memref<1xi32>) {
  %src, %dst = memref.distinct_objects %srcArg, %dstArg
    : memref<5xi32>, memref<1xi32>
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.for %j = 0 to 1 {
      affine.store %x0, %dst[0] : memref<1xi32>
    }
  }
  return
}

// A nested write that may alias the source must still block the transform.

// IR-LABEL: func.func @nested_same_source_write
// IR-NEXT: affine.for %[[I:.*]] = 0 to 4 {
// IR-NOT: iter_args
func.func @nested_same_source_write(%src: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.for %j = 0 to 1 {
      affine.store %x0, %src[0] : memref<5xi32>
    }
  }
  return
}

// Unknown effects nested under the container remain conservative.

// IR-LABEL: func.func @nested_unknown
// IR-NEXT: affine.for %[[I:.*]] = 0 to 4 {
// IR-NOT: iter_args
func.func @nested_unknown(%src: memref<5xi32>) {
  affine.for %i = 0 to 4 {
    %x0 = affine.load %src[%i] : memref<5xi32>
    %x1 = affine.load %src[%i + 1] : memref<5xi32>
    affine.for %j = 0 to 1 {
      func.call @unknown(%src) : (memref<5xi32>) -> ()
    }
  }
  return
}
