// RUN: circt-opt %s -ssp-schedule="scheduler=node-rl options=last-op-name=last,episodes=1,seed=0" | FileCheck %s --check-prefix=BASE
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule="scheduler=node-rl options=last-op-name=last,episodes=1,seed=0,local-search-nodes=8,local-search-time-limit=1" | FileCheck %s --check-prefix=LNS %}

// The greedy/learned resource ordering reaches the optimal II but leaves one
// cycle of objective latency. Re-optimizing only eight operations while the
// exterior schedule and reservations remain fixed closes that gap.
// BASE-LABEL: @local_search_covariance
// BASE-SAME: [II<16>]
// BASE: operation<@sink> @last({{.*}}) [t<16>]
// LNS-LABEL: @local_search_covariance
// LNS-SAME: [II<16>]
// LNS: operation<@sink> @last({{.*}}) [t<15>]
ssp.instance @local_search_covariance of "ModuloProblem" {
  library {
    operator_type @mul [latency<2>]
    operator_type @add [latency<1>]
    operator_type @div [latency<3>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @mul_r [limit<2>, ii<3>]
    resource_type @div_r [limit<1>, ii<2>]
  }
  graph {
    %mean0_0 = operation<@add> @mean0_0(@mean0_1 [dist<1>]) uses[@div_r]
    %norm0_0 = operation<@div>(%mean0_0) uses[@div_r]
    %cov0_0 = operation<@mul>(%norm0_0) uses[@mul_r]
    %mean0_1 = operation<@add> @mean0_1(%mean0_0) uses[@div_r]
    %norm0_1 = operation<@div>(%mean0_1) uses[@div_r]
    %cov0_1 = operation<@mul>(%norm0_1) uses[@mul_r]
    %mean1_0 = operation<@add> @mean1_0(@mean1_1 [dist<1>]) uses[@div_r]
    %norm1_0 = operation<@div>(%mean1_0) uses[@div_r]
    %cov1_0 = operation<@mul>(%norm1_0) uses[@mul_r]
    %mean1_1 = operation<@add> @mean1_1(%mean1_0) uses[@div_r]
    %norm1_1 = operation<@div>(%mean1_1) uses[@div_r]
    %cov1_1 = operation<@mul>(%norm1_1) uses[@mul_r]
    operation<@sink> @last(%cov0_1, %cov1_1)
  }
}
