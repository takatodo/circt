// RUN: not circt-opt %s -ssp-schedule="scheduler=simplex options=last-op-name=last" 2>&1 | FileCheck %s

// The modulo simplex resource heuristic cannot always preserve precedence when
// it expands the II. It must report the invalid candidate instead of returning
// it to the pass manager.
// CHECK: error: simplex modulo heuristic produced an invalid precedence schedule
ssp.instance @two_reductions of "ModuloProblem" {
  library {
    operator_type @mul [latency<2>]
    operator_type @add [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @mul_r [limit<2>, ii<1>]
  }
  graph {
    %m00 = operation<@mul>() uses[@mul_r]
    %a00 = operation<@add> @a00(%m00, @a02 [dist<1>]) uses[@mul_r]
    %m01 = operation<@mul>() uses[@mul_r]
    %a01 = operation<@add> @a01(%a00, %m01) uses[@mul_r]
    %m02 = operation<@mul>() uses[@mul_r]
    %a02 = operation<@add> @a02(%a01, %m02) uses[@mul_r]
    %m10 = operation<@mul>() uses[@mul_r]
    %a10 = operation<@add> @a10(%m10, @a12 [dist<1>]) uses[@mul_r]
    %m11 = operation<@mul>() uses[@mul_r]
    %a11 = operation<@add> @a11(%a10, %m11) uses[@mul_r]
    %m12 = operation<@mul>() uses[@mul_r]
    %a12 = operation<@add> @a12(%a11, %m12) uses[@mul_r]
    operation<@sink> @last(%a02, %a12)
  }
}
