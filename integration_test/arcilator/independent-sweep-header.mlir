// RUN: cat > %t.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <vector>
// RUN: #include "arcilator-independent-sweep.h"
// RUN:
// RUN: int main() {
// RUN:   constexpr std::size_t numStateBytes = 4;
// RUN:   constexpr std::size_t stride = numStateBytes;
// RUN:   std::vector<std::uint8_t> states(stride * 4, 0);
// RUN:   arcilator::IndependentSweepConfig config{4, 1, numStateBytes, stride,
// RUN:                                             states.size()};
// RUN:
// RUN:   auto fill = [](std::size_t stateIndex, std::uint8_t *state,
// RUN:                  std::size_t stateBytes) {
// RUN:     if (stateBytes < 4)
// RUN:       return;
// RUN:     state[0] = static_cast<std::uint8_t>(stateIndex);
// RUN:     state[1] = static_cast<std::uint8_t>(10 + stateIndex);
// RUN:     state[2] = 0;
// RUN:     state[3] = 0xaa;
// RUN:   };
// RUN:
// RUN:   auto eval = [](std::uint8_t *state, std::size_t stateBytes) {
// RUN:     if (stateBytes < 4)
// RUN:       return;
// RUN:     state[2] = static_cast<std::uint8_t>(state[0] + state[1]);
// RUN:   };
// RUN:
// RUN:   auto compare = [](std::size_t stateIndex, const std::uint8_t *state,
// RUN:                     std::size_t stateBytes) {
// RUN:     if (stateBytes < 4)
// RUN:       return false;
// RUN:     const auto expected =
// RUN:         static_cast<std::uint8_t>(stateIndex + 10 + stateIndex);
// RUN:     return state[2] == expected && state[3] == 0xaa;
// RUN:   };
// RUN:
// RUN:   const auto result =
// RUN:       arcilator::runIndependentSweep(states.data(), config, fill, eval,
// RUN:                                      compare);
// RUN:   if (result.caseCount != 4 || result.passedCount != 4 ||
// RUN:       result.failedCount != 0 || result.hasFailure)
// RUN:     return 1;
// RUN:   if (states[stride * 3 + 2] != 16)
// RUN:     return 2;
// RUN:   return 0;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator %t.cpp -o %t
// RUN: %t
