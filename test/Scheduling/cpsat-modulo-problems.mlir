// RUN: %if or-tools %{ circt-opt %s -ssp-schedule=scheduler=cpsat | FileCheck %s %}
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule="scheduler=cpsat options=time-limit=1,report-statistics=true" 2>&1 | FileCheck %s -check-prefix=STATS %}

// A small multi-resource recurrence whose exact optimum is II=3. This guards
// CP-SAT's modulo-phase encoding independently of the heuristic schedulers.
// CHECK-LABEL: @multi_resource_recurrence
// CHECK-SAME: [II<3>]
// STATS: cpsat: status=optimal, lower-bound=3, II=3
ssp.instance @multi_resource_recurrence of "ModuloProblem" {
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

// A resource hold may be much longer than its operation latency. CP-SAT's
// serial horizon must include the hold or these two valid schedules would be
// truncated at the latency-only horizon of three cycles.
// CHECK-LABEL: @long_hold_acyclic
ssp.instance @long_hold_acyclic of "SharedOperatorsProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @resource [limit<1>, ii<10>]
  }
  graph {
    %0 = operation<@op>() uses[@resource]
    %1 = operation<@op>() uses[@resource]
    // CHECK: operation<@sink> @last({{.*}}) [t<11>]
    operation<@sink> @last(%0, %1)
  }
}

// CHECK-LABEL: @long_hold_modulo
// CHECK-SAME: [II<20>]
// STATS: cpsat: status=optimal, lower-bound=20, II=20
ssp.instance @long_hold_modulo of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @resource [limit<1>, ii<10>]
  }
  graph {
    %0 = operation<@op>() uses[@resource]
    %1 = operation<@op>() uses[@resource]
    // CHECK: operation<@sink> @last({{.*}}) [t<11>]
    operation<@sink> @last(%0, %1)
  }
}
