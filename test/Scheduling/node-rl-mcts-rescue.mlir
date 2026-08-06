// RUN: circt-opt %s -ssp-schedule="scheduler=node-rl options=last-op-name=last,episodes=1,seed=0,mcts-trees=3,mcts-simulations=4,mcts-tree-width=8,mcts-rollout-width=8,mcts-time-limit=1" | FileCheck %s

// This fixed-II rescue case has loop-carried dependencies, two resources, and
// multi-cycle resource holds.  Its resource lower bound is 63; one ordinary
// NodeRL episode reaches II 67, while the bounded MCTS portfolio reaches 63.
// CHECK: ssp.instance @random_18 of "ModuloProblem" [II<63>]

ssp.instance @random_18 of "ModuloProblem" {
  library {
    operator_type @lat1 [latency<1>]
    operator_type @lat2 [latency<2>]
    operator_type @lat3 [latency<3>]
    operator_type @sink [latency<1>]
  }
  resource {
    resource_type @r0 [limit<2>, ii<2>]
    resource_type @r1 [limit<1>, ii<3>]
  }
  graph {
    %n0 = operation<@lat1> @n0(@n5 [dist<1>], @n33 [dist<3>]) uses[@r1]
    %n1 = operation<@lat1> @n1(@n0) uses[@r1]
    %n2 = operation<@lat3> @n2(@n1) uses[@r1]
    %n3 = operation<@lat2> @n3(@n2, @n0) uses[@r0]
    %n4 = operation<@lat2> @n4(@n3) uses[@r0]
    %n5 = operation<@lat1> @n5(@n4) uses[@r1]
    %n6 = operation<@lat1> @n6(@n11 [dist<3>], @n4, @n7 [dist<3>]) uses[@r0]
    %n7 = operation<@lat2> @n7(@n6) uses[@r0, @r1]
    %n8 = operation<@lat3> @n8(@n7, @n3) uses[@r0]
    %n9 = operation<@lat2> @n9(@n8, @n17 [dist<1>]) uses[@r0]
    %n10 = operation<@lat1> @n10(@n9) uses[@r0, @r1]
    %n11 = operation<@lat2> @n11(@n10, @n23 [dist<3>]) uses[@r0]
    %n12 = operation<@lat2> @n12(@n16 [dist<3>]) uses[@r1]
    %n13 = operation<@lat2> @n13(@n12, @n2) uses[@r0, @r1]
    %n14 = operation<@lat2> @n14(@n13) uses[@r0]
    %n15 = operation<@lat1> @n15(@n14) uses[@r0, @r1]
    %n16 = operation<@lat2> @n16(@n15, @n2, @n30 [dist<3>]) uses[@r0]
    %n17 = operation<@lat3> @n17(@n21 [dist<2>]) uses[@r0]
    %n18 = operation<@lat1> @n18(@n17, @n12) uses[@r0]
    %n19 = operation<@lat2> @n19(@n18) uses[@r0]
    %n20 = operation<@lat3> @n20(@n19) uses[@r0]
    %n21 = operation<@lat3> @n21(@n20) uses[@r0]
    %n22 = operation<@lat1> @n22(@n26 [dist<3>]) uses[@r0]
    %n23 = operation<@lat1> @n23(@n22) uses[@r1]
    %n24 = operation<@lat1> @n24(@n23, @n22) uses[@r0, @r1]
    %n25 = operation<@lat1> @n25(@n24, @n33 [dist<3>]) uses[@r1]
    %n26 = operation<@lat1> @n26(@n25, @n12) uses[@r1]
    %n27 = operation<@lat3> @n27(@n29 [dist<2>]) uses[@r1]
    %n28 = operation<@lat2> @n28(@n27, @n13) uses[@r0]
    %n29 = operation<@lat3> @n29(@n28, @n22) uses[@r1]
    %n30 = operation<@lat3> @n30(@n35 [dist<3>], @n22) uses[@r1]
    %n31 = operation<@lat1> @n31(@n30, @n10) uses[@r1]
    %n32 = operation<@lat3> @n32(@n31, @n19) uses[@r1]
    %n33 = operation<@lat2> @n33(@n32) uses[@r0]
    %n34 = operation<@lat1> @n34(@n33, @n1) uses[@r0]
    %n35 = operation<@lat3> @n35(@n34, @n39 [dist<3>], @n39 [dist<2>]) uses[@r0]
    %n36 = operation<@lat2> @n36(@n39 [dist<2>], @n10) uses[@r1]
    %n37 = operation<@lat2> @n37(@n36) uses[@r1]
    %n38 = operation<@lat1> @n38(@n37) uses[@r0]
    %n39 = operation<@lat2> @n39(@n38) uses[@r1]
    operation<@sink> @last(%n0, %n1, %n2, %n3, %n4, %n5, %n6, %n7, %n8, %n9, %n10, %n11, %n12, %n13, %n14, %n15, %n16, %n17, %n18, %n19, %n20, %n21, %n22, %n23, %n24, %n25, %n26, %n27, %n28, %n29, %n30, %n31, %n32, %n33, %n34, %n35, %n36, %n37, %n38, %n39)
  }
}
