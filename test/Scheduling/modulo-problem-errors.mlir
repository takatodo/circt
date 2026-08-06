// RUN: circt-opt %s -ssp-roundtrip=verify -verify-diagnostics -split-input-file

// expected-error@+1 {{Resource type 'limited_rsrc' is oversubscribed}}
ssp.instance @oversubscribed of "ModuloProblem" [II<2>] {
  library {
    operator_type @limited [latency<1>]
  }
  resource {
    resource_type @limited_rsrc [limit<2>]
  }
  graph {
    operation<@limited>() uses[@limited_rsrc] [t<1>]
    operation<@limited>() uses[@limited_rsrc] [t<3>]
    operation<@limited>() uses[@limited_rsrc] [t<5>]
  }
}

// -----

// One operation can overlap with a request from its next iteration.
// expected-error@+1 {{Resource type 'limited_rsrc' is oversubscribed}}
ssp.instance @self_overlapping_reservation of "ModuloProblem" [II<2>] {
  library {
    operator_type @limited [latency<1>]
  }
  resource {
    resource_type @limited_rsrc [limit<1>, ii<3>]
  }
  graph {
    operation<@limited>() uses[@limited_rsrc] [t<0>]
  }
}
