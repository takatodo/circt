// RUN: circt-opt %s -ssp-roundtrip | FileCheck %s

// CHECK-LABEL: @resource_cost
ssp.instance @resource_cost of "SharedOperatorsProblem" {
  library {
    operator_type @mul [latency<3>]
    operator_type @sink [latency<1>]
  }
  resource {
    // CHECK: resource_type @mul_unit [limit<2>, ii<3>, cost<7>]
    resource_type @mul_unit [limit<2>, ii<3>, cost<7>]
  }
  graph {
    %0 = operation<@mul>() uses[@mul_unit]
    operation<@sink> @last(%0)
  }
}
