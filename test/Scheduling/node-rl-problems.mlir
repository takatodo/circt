// RUN: circt-opt %s -ssp-schedule="scheduler=node-rl options=last-op-name=last,episodes=32,seed=7,learning-rate=0.05,exploration=1.0" | FileCheck %s
// RUN: circt-opt %s -ssp-schedule="scheduler=rl options=last-op-name=last,episodes=32,seed=7" | FileCheck %s

// A source-order list scheduler assigns the shared resource to %unrelated
// first, delaying the critical path by one cycle.  The node policy recognizes
// that %critical reaches the objective and prioritizes it instead.
// CHECK-LABEL: @critical_priority
ssp.instance @critical_priority of "SharedOperatorsProblem" {
  library {
    operator_type @shared [latency<1>]
    operator_type @long [latency<4>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @alu [limit<1>]
  }
  graph {
    // CHECK: %[[UNRELATED:.+]] = operation<@shared>() uses[@alu] [t<1>, bindings<[0]>]
    %unrelated = operation<@shared>() uses[@alu]
    // CHECK: %[[CRITICAL:.+]] = operation<@shared>() uses[@alu] [t<0>, bindings<[0]>]
    %critical = operation<@shared>() uses[@alu]
    // CHECK: %[[CHAIN:.+]] = operation<@long>(%[[CRITICAL]]) [t<1>]
    %chain = operation<@long>(%critical)
    // CHECK: operation<@sink> @last(%[[CHAIN]]) [t<5>]
    operation<@sink> @last(%chain)
  }
}

// A node may reserve several independently limited resources in the same
// cycle.  The existing simplex heuristic only handles one resource per node.
// CHECK-LABEL: @multiple_resources
ssp.instance @multiple_resources of "SharedOperatorsProblem" {
  library {
    operator_type @both [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @left [limit<1>]
    resource_type @right [limit<1>]
  }
  graph {
    // CHECK: %[[FIRST:.+]] = operation<@both>() uses[@left, @right] [t<0>, bindings<[0, 0]>]
    %first = operation<@both>() uses[@left, @right]
    // CHECK: %[[SECOND:.+]] = operation<@both>() uses[@left, @right] [t<1>, bindings<[0, 0]>]
    %second = operation<@both>() uses[@left, @right]
    // CHECK: operation<@sink> @last(%[[FIRST]], %[[SECOND]]) [t<2>]
    operation<@sink> @last(%first, %second)
  }
}

// Resource initiation intervals model non-fully-pipelined resources. The
// first operation holds @right for three cycles, so the second one cannot
// start before t=3 even though both operations have latency one.
// CHECK-LABEL: @resource_initiation_intervals
ssp.instance @resource_initiation_intervals of "SharedOperatorsProblem" {
  library {
    operator_type @both [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @left [limit<1>, ii<2>]
    resource_type @right [limit<1>, ii<3>]
  }
  graph {
    // CHECK: %[[FIRST:.+]] = operation<@both>() uses[@left, @right] [t<0>, bindings<[0, 0]>]
    %first = operation<@both>() uses[@left, @right]
    // CHECK: %[[SECOND:.+]] = operation<@both>() uses[@left, @right] [t<3>, bindings<[0, 0]>]
    %second = operation<@both>() uses[@left, @right]
    // CHECK: operation<@sink> @last(%[[FIRST]], %[[SECOND]]) [t<4>]
    operation<@sink> @last(%first, %second)
  }
}

// A loop-carried dependence is solved as a cyclic difference constraint.  The
// non-pipelined multiplier is reserved for two modulo slots per request, so
// the three requests require II=6 on one physical instance.
// CHECK-LABEL: @loop_carried_modulo
// CHECK-SAME: [II<6>]
ssp.instance @loop_carried_modulo of "ModuloProblem" {
  library {
    operator_type @mul [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @mul_rsrc [limit<1>, ii<2>]
  }
  graph {
    // CHECK: %[[A:.+]] = operation<@mul>(@b [dist<1>]) uses[@mul_rsrc] [t<0>, bindings<[0]>]
    %a = operation<@mul>(@b [dist<1>]) uses[@mul_rsrc]
    // CHECK: %[[B:.+]] = operation<@mul> @b(%[[A]]) uses[@mul_rsrc] [t<2>, bindings<[0]>]
    %b = operation<@mul> @b(%a) uses[@mul_rsrc]
    // CHECK: %[[C:.+]] = operation<@mul>() uses[@mul_rsrc] [t<4>, bindings<[0]>]
    %c = operation<@mul>() uses[@mul_rsrc]
    // CHECK: operation<@sink> @last(%[[A]], %[[B]], %[[C]]) [t<5>]
    operation<@sink> @last(%a, %b, %c)
  }
}

// One operation is issued every II=2 cycles but reserves a multiplier for
// three cycles. Two instances suffice by rotating the binding across loop
// iterations. The aggregate modulo reservation is valid, but the static
// per-operation `bindings` property is intentionally omitted.
// CHECK-LABEL: @rotating_resource_binding
// CHECK-SAME: [II<2>]
ssp.instance @rotating_resource_binding of "ModuloProblem" {
  library {
    operator_type @mul [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @mul_rsrc [limit<2>, ii<3>]
  }
  graph {
    // CHECK: %[[OP:.+]] = operation<@mul> @op(@op [dist<1>]) uses[@mul_rsrc] [t<0>]
    %op = operation<@mul> @op(@op [dist<1>]) uses[@mul_rsrc]
    // CHECK: operation<@sink> @last(%[[OP]]) [t<1>]
    operation<@sink> @last(%op)
  }
}

// A later consumer initially aliases the recurrence operation's phase even
// though their absolute starts differ by one II. A signed resource-ordering
// offset moves the consumer to the next phase without unnecessarily raising
// the recurrence-bound II.
// CHECK-LABEL: @fully_pipelined_multi_resource_recurrence
// CHECK-SAME: [II<3>]
ssp.instance @fully_pipelined_multi_resource_recurrence of "ModuloProblem" {
  library {
    operator_type @mul [latency<2>]
    operator_type @div [latency<3>]
    operator_type @join [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @mul_r [limit<2>, ii<1>]
    resource_type @div_r [limit<1>, ii<1>]
  }
  graph {
    %m = operation<@mul> @m(@m [dist<1>]) uses[@mul_r]
    %d = operation<@div> @d(@d [dist<1>]) uses[@div_r]
    %j = operation<@join>(%m, %d) uses[@mul_r, @div_r]
    operation<@sink> @last(%j)
  }
}

// Users of @left and @right have no shared resource and their dependence
// components meet only at the sink. They can be modulo-scheduled as separate
// clusters and then shifted along the acyclic inter-cluster dependences.
// CHECK-LABEL: @independent_resource_clusters
// CHECK-SAME: [II<6>]
ssp.instance @independent_resource_clusters of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @left [limit<1>, ii<2>]
    resource_type @right [limit<1>, ii<3>]
  }
  graph {
    // CHECK: %[[A:.+]] = operation<@op> @a(@a [dist<1>]) uses[@left] [t<0>, bindings<[0]>]
    %a = operation<@op> @a(@a [dist<1>]) uses[@left]
    // CHECK: %[[B:.+]] = operation<@op>() uses[@left] [t<2>, bindings<[0]>]
    %b = operation<@op>() uses[@left]
    // CHECK: %[[C:.+]] = operation<@op> @c(@c [dist<1>]) uses[@right] [t<0>, bindings<[0]>]
    %c = operation<@op> @c(@c [dist<1>]) uses[@right]
    // CHECK: %[[D:.+]] = operation<@op>() uses[@right] [t<3>, bindings<[0]>]
    %d = operation<@op>() uses[@right]
    // CHECK: operation<@sink> @last(%[[A]], %[[B]], %[[C]], %[[D]]) [t<4>]
    operation<@sink> @last(%a, %b, %c, %d)
  }
}

// Joining %a and %c through @shared creates a cycle in the resource-group
// quotient (%a/%c -> %b -> %a/%c). SCC closure must put all three operations
// back into one scheduling cluster before constructing the cluster DAG.
// CHECK-LABEL: @resource_quotient_scc
// CHECK-SAME: [II<2>]
ssp.instance @resource_quotient_scc of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @shared [limit<1>]
  }
  graph {
    // CHECK: %[[A:.+]] = operation<@op>() uses[@shared] [t<0>, bindings<[0]>]
    %a = operation<@op>() uses[@shared]
    // CHECK: %[[B:.+]] = operation<@op>(%[[A]]) [t<1>]
    %b = operation<@op>(%a)
    // CHECK: %[[C:.+]] = operation<@op>(%[[B]]) uses[@shared] [t<3>, bindings<[0]>]
    %c = operation<@op>(%b) uses[@shared]
    // CHECK: operation<@sink> @last(%[[C]]) [t<4>]
    operation<@sink> @last(%c)
  }
}

// A high-capacity pool is partitioned into virtual ordering lanes inside one
// episode. Sixteen two-cycle reservations on four instances attain their
// resource lower bound II=8 without rebuilding the complete reservation table
// after every ordering edge.
// CHECK-LABEL: @bulk_high_capacity_resource
// CHECK-SAME: [II<8>]
ssp.instance @bulk_high_capacity_resource of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @pool [limit<4>, ii<2>]
  }
  graph {
    %0 = operation<@op>() uses[@pool]
    %1 = operation<@op>() uses[@pool]
    %2 = operation<@op>() uses[@pool]
    %3 = operation<@op>() uses[@pool]
    %4 = operation<@op>() uses[@pool]
    %5 = operation<@op>() uses[@pool]
    %6 = operation<@op>() uses[@pool]
    %7 = operation<@op>() uses[@pool]
    %8 = operation<@op>() uses[@pool]
    %9 = operation<@op>() uses[@pool]
    %10 = operation<@op>() uses[@pool]
    %11 = operation<@op>() uses[@pool]
    %12 = operation<@op>() uses[@pool]
    %13 = operation<@op>() uses[@pool]
    %14 = operation<@op>() uses[@pool]
    %15 = operation<@op>() uses[@pool]
    // CHECK: operation<@sink> @last({{.*}}) [t<7>]
    operation<@sink> @last(%0, %1, %2, %3, %4, %5, %6, %7,
                              %8, %9, %10, %11, %12, %13, %14, %15)
  }
}

// A low-capacity resource with dozens of users also needs bulk ordering: the
// individual overflow buckets are small, but repairing them pairwise consumes
// a quadratic number of ordering attempts. The internal lanes are a safe
// restriction of the original two-instance pool and attain its II bound.
// CHECK-LABEL: @bulk_large_low_capacity_resource
// CHECK-SAME: [II<32>]
ssp.instance @bulk_large_low_capacity_resource of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @pool [limit<2>, ii<2>]
  }
  graph {
    %0 = operation<@op>() uses[@pool]
    %1 = operation<@op>() uses[@pool]
    %2 = operation<@op>() uses[@pool]
    %3 = operation<@op>() uses[@pool]
    %4 = operation<@op>() uses[@pool]
    %5 = operation<@op>() uses[@pool]
    %6 = operation<@op>() uses[@pool]
    %7 = operation<@op>() uses[@pool]
    %8 = operation<@op>() uses[@pool]
    %9 = operation<@op>() uses[@pool]
    %10 = operation<@op>() uses[@pool]
    %11 = operation<@op>() uses[@pool]
    %12 = operation<@op>() uses[@pool]
    %13 = operation<@op>() uses[@pool]
    %14 = operation<@op>() uses[@pool]
    %15 = operation<@op>() uses[@pool]
    %16 = operation<@op>() uses[@pool]
    %17 = operation<@op>() uses[@pool]
    %18 = operation<@op>() uses[@pool]
    %19 = operation<@op>() uses[@pool]
    %20 = operation<@op>() uses[@pool]
    %21 = operation<@op>() uses[@pool]
    %22 = operation<@op>() uses[@pool]
    %23 = operation<@op>() uses[@pool]
    %24 = operation<@op>() uses[@pool]
    %25 = operation<@op>() uses[@pool]
    %26 = operation<@op>() uses[@pool]
    %27 = operation<@op>() uses[@pool]
    %28 = operation<@op>() uses[@pool]
    %29 = operation<@op>() uses[@pool]
    %30 = operation<@op>() uses[@pool]
    %31 = operation<@op>() uses[@pool]
    // CHECK: operation<@sink> @last({{.*}}) [t<31>]
    operation<@sink> @last(%0, %1, %2, %3, %4, %5, %6, %7,
                              %8, %9, %10, %11, %12, %13, %14, %15,
                              %16, %17, %18, %19, %20, %21, %22, %23,
                              %24, %25, %26, %27, %28, %29, %30, %31)
  }
}

// The joins couple two otherwise independent held-resource groups. Pairwise
// repair cannot coordinate both saturated reservation tables at their exact
// resource bound; bulk ordering must use one consistent policy ranking for
// both groups rather than raising II from 12 to 13.
// CHECK-LABEL: @coupled_multi_resource_bulk_ordering
// CHECK-SAME: [II<12>]
ssp.instance @coupled_multi_resource_bulk_ordering of "ModuloProblem" {
  library {
    operator_type @mul [latency<2>]
    operator_type @div [latency<3>]
    operator_type @join [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @mul_r [limit<2>, ii<3>]
    resource_type @div_r [limit<2>, ii<2>]
  }
  graph {
    %m0_0 = operation<@mul> @m0_0(@m0_0 [dist<1>]) uses[@mul_r]
    %d0_0 = operation<@div> @d0_0(@d0_0 [dist<1>]) uses[@div_r]
    %m0_1 = operation<@mul>(%m0_0) uses[@mul_r]
    %d0_1 = operation<@div>(%d0_0) uses[@div_r]
    %m0_2 = operation<@mul>(%m0_1) uses[@mul_r]
    %d0_2 = operation<@div>(%d0_1) uses[@div_r]
    %j0 = operation<@join>(%m0_2, %d0_2) uses[@mul_r, @div_r]
    %m1_0 = operation<@mul> @m1_0(@m1_0 [dist<1>]) uses[@mul_r]
    %d1_0 = operation<@div> @d1_0(@d1_0 [dist<1>]) uses[@div_r]
    %m1_1 = operation<@mul>(%m1_0) uses[@mul_r]
    %d1_1 = operation<@div>(%d1_0) uses[@div_r]
    %m1_2 = operation<@mul>(%m1_1) uses[@mul_r]
    %d1_2 = operation<@div>(%d1_1) uses[@div_r]
    %j1 = operation<@join>(%m1_2, %d1_2) uses[@mul_r, @div_r]
    // CHECK: operation<@sink> @last({{.*}}) [t<10>]
    operation<@sink> @last(%j0, %j1)
  }
}
