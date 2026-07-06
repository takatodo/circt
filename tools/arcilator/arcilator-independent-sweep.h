// NOLINTBEGIN
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace arcilator {

struct IndependentSweepConfig {
  std::size_t stateCount = 0;
  std::size_t steps = 0;
  std::size_t numStateBytes = 0;
  std::size_t slotStride = 0;
  std::size_t stateStorageBytes = 0;
};

struct IndependentSweepResult {
  std::size_t caseCount = 0;
  std::size_t passedCount = 0;
  std::size_t failedCount = 0;
  std::size_t firstFailedCase = 0;
  bool hasFailure = false;
};

inline bool isValid(const IndependentSweepConfig &config) {
  if (config.stateCount == 0 || config.steps == 0 ||
      config.numStateBytes == 0 || config.slotStride < config.numStateBytes)
    return false;
  if (config.stateCount >
      std::numeric_limits<std::size_t>::max() / config.slotStride)
    return false;
  return config.stateStorageBytes >= config.stateCount * config.slotStride;
}

inline std::uint8_t *stateSlot(std::uint8_t *states,
                               const IndependentSweepConfig &config,
                               std::size_t stateIndex) {
  return states + stateIndex * config.slotStride;
}

inline const std::uint8_t *stateSlot(const std::uint8_t *states,
                                     const IndependentSweepConfig &config,
                                     std::size_t stateIndex) {
  return states + stateIndex * config.slotStride;
}

template <typename FillFn, typename EvalFn, typename CompareFn>
IndependentSweepResult
runIndependentSweep(std::uint8_t *states, const IndependentSweepConfig &config,
                    FillFn fill, EvalFn eval, CompareFn compare) {
  IndependentSweepResult result;
  result.caseCount = config.stateCount;

  if (states == nullptr || !isValid(config)) {
    result.failedCount = config.stateCount;
    result.hasFailure = true;
    return result;
  }

  for (std::size_t stateIndex = 0; stateIndex < config.stateCount;
       ++stateIndex) {
    std::uint8_t *slot = stateSlot(states, config, stateIndex);
    fill(stateIndex, slot, config.numStateBytes);

    for (std::size_t step = 0; step < config.steps; ++step)
      eval(slot, config.numStateBytes);

    if (compare(stateIndex, slot, config.numStateBytes)) {
      ++result.passedCount;
      continue;
    }

    if (!result.hasFailure) {
      result.firstFailedCase = stateIndex;
      result.hasFailure = true;
    }
    ++result.failedCount;
  }

  return result;
}

} // namespace arcilator

// NOLINTEND
