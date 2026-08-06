// RUN: %if or-tools %{ circt-opt %s -ssp-schedule="scheduler=cpsat options=time-limit=2,workers=1,minimize-latency=false,resource-model=auto,report-statistics=true" -ssp-roundtrip=verify 2>&1 | FileCheck %s %}

// A small operation graph can still have a prohibitively large direct modulo
// encoding when a resource reservation spans many phases. Auto must select the
// exact cumulative model before creating 80,000 reified phase indicators.
// CHECK: cpsat: status=optimal, lower-bound=200, II=200, resource-model=cumulative, phase-indicators=80000
ssp.instance @auto_cumulative of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @held [limit<2>, ii<200>]
  }
  graph {
    %first = operation<@op>() uses[@held]
    %second = operation<@op>() uses[@held]
    operation<@sink> @last(%first, %second)
  }
}
