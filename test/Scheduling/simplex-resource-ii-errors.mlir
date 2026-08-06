// RUN: circt-opt %s -ssp-schedule=scheduler=simplex -verify-diagnostics -split-input-file

// expected-error@+1 {{simplex scheduling does not support resource initiation intervals greater than one}}
ssp.instance @resource_ii of "SharedOperatorsProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @resource [limit<1>, ii<2>]
  }
  graph {
    %0 = operation<@op>() uses[@resource]
    operation<@sink> @last(%0)
  }
}

// -----

ssp.instance @multiple_resources of "SharedOperatorsProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @left [limit<1>]
    resource_type @right [limit<1>]
  }
  graph {
    // expected-error@+1 {{simplex scheduling does not support operations using multiple limited resources}}
    %0 = operation<@op>() uses[@left, @right]
    operation<@sink> @last(%0)
  }
}
