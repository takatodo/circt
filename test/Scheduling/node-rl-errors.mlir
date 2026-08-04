// RUN: circt-opt %s -ssp-schedule="scheduler=node-rl options=episodes=8" -verify-diagnostics -split-input-file

// expected-error@+1 {{dependence cycle detected}}
ssp.instance @cyclic_graph of "SharedOperatorsProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  resource {
    resource_type @alu [limit<1>]
  }
  graph {
    %0 = operation<@_1>(@op2) uses[@alu]
    %1 = operation<@_1>(%0) uses[@alu]
    operation<@_1> @op2(%1) uses[@alu]
  }
}
