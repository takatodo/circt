// RUN: circt-opt %s -ssp-schedule="scheduler=node-rl options=episodes=0" -verify-diagnostics
// RUN: not circt-opt %s -ssp-schedule="scheduler=node-rl options=episode-node-budget=bad" 2>&1 | FileCheck %s --check-prefix=BUDGET
// RUN: not circt-opt %s -ssp-schedule="scheduler=node-rl options=resource-ordering-budget=bad" 2>&1 | FileCheck %s --check-prefix=ORDERING-BUDGET
// RUN: not circt-opt %s -ssp-schedule="scheduler=node-rl options=local-search-nodes=bad" 2>&1 | FileCheck %s --check-prefix=LOCAL-NODES
// RUN: not circt-opt %s -ssp-schedule="scheduler=node-rl options=local-search-time-limit=0" 2>&1 | FileCheck %s --check-prefix=LOCAL-TIME

// BUDGET: invalid node-rl 'episode-node-budget' option; expected an unsigned integer
// ORDERING-BUDGET: invalid node-rl 'resource-ordering-budget' option; expected an unsigned integer
// LOCAL-NODES: invalid node-rl 'local-search-nodes' option; expected an unsigned integer
// LOCAL-TIME: invalid node-rl 'local-search-time-limit' option; expected a finite positive number

// expected-error@+1 {{invalid node-rl 'episodes' option; expected a positive integer}}
ssp.instance @invalid_episodes of "SharedOperatorsProblem" {
  library {
    operator_type @_1 [latency<1>]
  }
  resource {
  }
  graph {
    operation<@_1>()
  }
}
