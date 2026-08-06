//===- ResourceSchedule.h - Resource schedule certificates ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_LIB_SCHEDULING_RESOURCESCHEDULE_H
#define CIRCT_LIB_SCHEDULING_RESOURCESCHEDULE_H

#include "circt/Scheduling/Algorithms.h"

namespace circt {
namespace scheduling {
namespace detail {

using ResourceScheduleFunction = function_ref<LogicalResult()>;

FailureOr<SmallVector<ResourceParetoPoint>> exploreResourcePareto(
    SharedOperatorsProblem &prob, Operation *lastOp,
    ArrayRef<ResourceAllocation> allocations, ResourceCostFunction costFunction,
    ResourceScheduleFunction schedule, StringRef schedulerName);

FailureOr<SmallVector<ResourceParetoPoint>> exploreResourcePareto(
    ModuloProblem &prob, Operation *lastOp,
    ArrayRef<ResourceAllocation> allocations, ResourceCostFunction costFunction,
    ResourceScheduleFunction schedule, StringRef schedulerName);

} // namespace detail
} // namespace scheduling
} // namespace circt

#endif // CIRCT_LIB_SCHEDULING_RESOURCESCHEDULE_H
