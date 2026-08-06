// RUN: not circt-opt %s -ssp-schedule="scheduler=node-rl options=last-op-name=last,episodes=8,resource-ordering-budget=1" 2>&1 | FileCheck %s

// CHECK: node-rl exhausted its resource-ordering budget of 1 before finding a feasible schedule
ssp.instance @ordering_budget of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @shared [limit<1>]
  }
  graph {
    %0 = operation<@op>() uses[@shared]
    %1 = operation<@op>() uses[@shared]
    %2 = operation<@op>() uses[@shared]
    operation<@sink> @last(%0, %1, %2)
  }
}
