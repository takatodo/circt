// RUN: circt-opt %s -ssp-schedule="scheduler=simplex options=last-op-name=last" -verify-diagnostics -split-input-file

// expected-error@+1 {{last operation is not a sink}}
ssp.instance of "ModuloProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  resource {
  }
  graph {
    operation<@_1> @last()
    operation<@_1>(@last)
  }
}

// -----

// expected-error@+1 {{simplex modulo scheduling does not support resource initiation intervals greater than one}}
ssp.instance @resource_ii of "ModuloProblem" {
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

// expected-error@+1 {{multiple sinks detected}}
ssp.instance of "ModuloProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  resource {
  }
  graph {
    operation<@_1>()
    operation<@_1> @last()
  }
}
