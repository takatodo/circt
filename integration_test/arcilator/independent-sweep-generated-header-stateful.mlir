// REQUIRES: python
//
// RUN: mkdir -p %t.dir
// RUN: cat > %t.dir/state.json <<'EOF'
// RUN: [
// RUN:   {
// RUN:     "name": "StatefulToy",
// RUN:     "numStateBytes": 16,
// RUN:     "initialFnSym": "StatefulToy_initial",
// RUN:     "states": [
// RUN:       {"name": "inp", "offset": 0, "numBits": 16, "type": "input"},
// RUN:       {"name": "out", "offset": 2, "numBits": 16, "type": "output"},
// RUN:       {"name": "core/acc", "offset": 4, "numBits": 16, "type": "register"},
// RUN:       {"name": "core/count", "offset": 6, "numBits": 8, "type": "register"},
// RUN:       {"name": "core/taps", "offset": 8, "numBits": 16, "type": "memory", "stride": 2, "depth": 2}
// RUN:     ]
// RUN:   }
// RUN: ]
// RUN: EOF
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/state.json > %t.dir/stateful_toy_model.h
// RUN: cat > %t.dir/test.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <cstring>
// RUN: #include <string>
// RUN: #include <vector>
// RUN: #include "arcilator-independent-sweep.h"
// RUN: #include "stateful_toy_model.h"
// RUN:
// RUN: extern "C" void StatefulToy_initial(void *rawState) {
// RUN:   std::memset(rawState, 0, StatefulToyLayout::numStateBytes);
// RUN: }
// RUN:
// RUN: extern "C" void StatefulToy_eval(void *rawState) {
// RUN:   StatefulToyView view(static_cast<std::uint8_t *>(rawState));
// RUN:   const auto tap =
// RUN:       view.internal.core.taps.words[view.internal.core.count & 1].data;
// RUN:   view.internal.core.acc =
// RUN:       static_cast<std::uint16_t>(view.internal.core.acc + view.inp + tap);
// RUN:   view.internal.core.count =
// RUN:       static_cast<std::uint8_t>(view.internal.core.count + 1);
// RUN:   view.out = view.internal.core.acc;
// RUN: }
// RUN:
// RUN: int main() {
// RUN:   constexpr std::size_t stateCount = 5;
// RUN:   constexpr std::size_t stride = StatefulToyLayout::numStateBytes;
// RUN:   std::vector<std::uint8_t> states(stride * stateCount, 0);
// RUN:   arcilator::IndependentSweepConfig config{
// RUN:       stateCount, 3, StatefulToyLayout::numStateBytes, stride,
// RUN:       states.size()};
// RUN:
// RUN:   auto fill = [](std::size_t stateIndex, std::uint8_t *state,
// RUN:                  std::size_t) {
// RUN:     StatefulToy_initial(state);
// RUN:     StatefulToyView view(state);
// RUN:     view.inp = static_cast<std::uint16_t>(10 + stateIndex);
// RUN:     view.internal.core.taps.words[0].data = 3;
// RUN:     view.internal.core.taps.words[1].data = 5;
// RUN:   };
// RUN:
// RUN:   auto eval = [](std::uint8_t *state, std::size_t) {
// RUN:     StatefulToy_eval(state);
// RUN:   };
// RUN:
// RUN:   auto compare = [](std::size_t stateIndex, const std::uint8_t *state,
// RUN:                     std::size_t) {
// RUN:     StatefulToyView view(const_cast<std::uint8_t *>(state));
// RUN:     const auto inp = static_cast<std::uint16_t>(10 + stateIndex);
// RUN:     const auto expected = static_cast<std::uint16_t>(3 * inp + 11);
// RUN:     return view.out == expected && view.internal.core.acc == expected &&
// RUN:            view.internal.core.count == 3 &&
// RUN:            view.internal.core.taps.words[0].data == 3 &&
// RUN:            view.internal.core.taps.words[1].data == 5;
// RUN:   };
// RUN:
// RUN:   const auto result =
// RUN:       arcilator::runIndependentSweep(states.data(), config, fill, eval,
// RUN:                                      compare);
// RUN:   if (result.caseCount != stateCount || result.passedCount != stateCount ||
// RUN:       result.failedCount != 0 || result.hasFailure)
// RUN:     return 1;
// RUN:
// RUN:   StatefulToy single;
// RUN:   single.view.inp = 11;
// RUN:   single.view.internal.core.taps.words[0].data = 3;
// RUN:   single.view.internal.core.taps.words[1].data = 5;
// RUN:   single.eval();
// RUN:   single.eval();
// RUN:   single.eval();
// RUN:   if (single.view.out != 44 || single.view.internal.core.acc != 44 ||
// RUN:       single.view.internal.core.count != 3)
// RUN:     return 2;
// RUN:
// RUN:   StatefulToyView first(states.data());
// RUN:   StatefulToyView last(states.data() + stride * (stateCount - 1));
// RUN:   return first.out != last.out && first.internal.core.count == 3 &&
// RUN:                  last.internal.core.count == 3
// RUN:              ? 0
// RUN:              : 3;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator -I %t.dir %t.dir/test.cpp -o %t.dir/test
// RUN: %t.dir/test
