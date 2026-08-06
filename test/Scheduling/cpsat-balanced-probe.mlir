// RUN: %if or-tools %{ circt-opt %s -ssp-schedule="scheduler=cpsat options=workers=2,minimize-latency=false,balanced-probe=true,balanced-probe-time-limit=1,report-statistics=true" -ssp-roundtrip=verify 2>&1 | FileCheck %s %}
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule="scheduler=cpsat options=workers=2,minimize-latency=true,balanced-probe=true,balanced-probe-time-limit=1,report-statistics=true" -ssp-roundtrip=verify 2>&1 | FileCheck %s -check-prefix=LATENCY %}
// RUN: %if or-tools %{ circt-opt %s -ssp-schedule="scheduler=cpsat options=workers=2,minimize-latency=true,resource-model=cumulative,balanced-probe=true,balanced-probe-time-limit=1,report-statistics=true" -ssp-roundtrip=verify 2>&1 | FileCheck %s -check-prefix=LATENCY-CUM %}
// LATENCY: cpsat: status=optimal, lower-bound=2, II=2{{.*}}balanced-probe=hit
// LATENCY-NEXT: cpsat: status=optimal, lower-bound=1, II=1{{.*}}balanced-probe=skipped
// LATENCY-NEXT: cpsat: status=optimal, lower-bound=5, II=5{{.*}}balanced-probe=miss
// LATENCY: operation<@sink> @last({{.*}}) [t<5>]
// LATENCY-CUM: cpsat: status=optimal, lower-bound=2, II=2, resource-model=cumulative{{.*}}balanced-probe=hit
// LATENCY-CUM-NEXT: cpsat: status=optimal, lower-bound=1, II=1, resource-model=cumulative{{.*}}balanced-probe=skipped
// LATENCY-CUM-NEXT: cpsat: status=optimal, lower-bound=5, II=5, resource-model=cumulative{{.*}}balanced-probe=miss
// LATENCY-CUM: operation<@sink> @last({{.*}}) [t<5>]

// The two resource users must occupy distinct phases at II=2, but the
// latency-two dependence prevents both starts from fitting in [0, II).  This
// forces the balanced probe's single-stage member to fail and its absolute
// stage member to produce the verified schedule.
// CHECK: cpsat: status=optimal, lower-bound=2, II=2{{.*}}balanced-probe=hit
ssp.instance @absolute_stage_probe of "ModuloProblem" {
  library {
    operator_type @lat2 [latency<2>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @resource [limit<1>, ii<1>]
  }
  graph {
    %a = operation<@lat2>() uses[@resource]
    %b = operation<@lat2>(%a) uses[@resource]
    operation<@sink> @last(%b)
  }
}

// More than five limited resources bypass the deliberately narrow probe and
// retain the complete CP-SAT path.
// CHECK: cpsat: status=optimal, lower-bound=1, II=1{{.*}}balanced-probe=skipped
ssp.instance @resource_count_gate of "ModuloProblem" {
  library {
    operator_type @op [latency<1>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @r0 [limit<1>, ii<1>]
    resource_type @r1 [limit<1>, ii<1>]
    resource_type @r2 [limit<1>, ii<1>]
    resource_type @r3 [limit<1>, ii<1>]
    resource_type @r4 [limit<1>, ii<1>]
    resource_type @r5 [limit<1>, ii<1>]
  }
  graph {
    %op = operation<@op>() uses[@r0, @r1, @r2, @r3, @r4, @r5]
    operation<@sink> @last(%op)
  }
}

// The recurrence fixes b exactly one cycle after a at II=5.  The unrestricted
// resource model accepts those adjacent phases, whereas the two-user balanced
// pattern only permits phase distances two and three.  A restricted miss must
// therefore fall through to the complete model without raising II.
// CHECK: cpsat: status=optimal, lower-bound=5, II=5{{.*}}balanced-probe=miss
ssp.instance @complete_fallback of "ModuloProblem" {
  library {
    operator_type @lat1 [latency<1>]
    operator_type @lat4 [latency<4>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @resource [limit<1>, ii<1>]
  }
  graph {
    %a = operation<@lat1> @a(@b [dist<1>]) uses[@resource]
    %b = operation<@lat4> @b(%a) uses[@resource]
    operation<@sink> @last(%b)
  }
}
