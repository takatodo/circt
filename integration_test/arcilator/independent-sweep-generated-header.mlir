// REQUIRES: python
//
// RUN: mkdir -p %t.dir
// RUN: cat > %t.dir/state.json <<'EOF'
// RUN: [
// RUN:   {
// RUN:     "name": "Toy",
// RUN:     "numStateBytes": 4,
// RUN:     "initialFnSym": "Toy_initial",
// RUN:     "states": [
// RUN:       {"name": "a", "offset": 0, "numBits": 8, "type": "input"},
// RUN:       {"name": "b", "offset": 1, "numBits": 8, "type": "input"},
// RUN:       {"name": "sum", "offset": 2, "numBits": 8, "type": "output"},
// RUN:       {"name": "scratch", "offset": 3, "numBits": 8, "type": "register"}
// RUN:     ]
// RUN:   }
// RUN: ]
// RUN: EOF
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/state.json > %t.dir/toy_model.h
// RUN: cat > %t.dir/test.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <cstring>
// RUN: #include <string>
// RUN: #include <vector>
// RUN: #include "arcilator-independent-sweep.h"
// RUN: #include "toy_model.h"
// RUN:
// RUN: extern "C" void Toy_initial(void *rawState) {
// RUN:   std::memset(rawState, 0, ToyLayout::numStateBytes);
// RUN: }
// RUN:
// RUN: extern "C" void Toy_eval(void *rawState) {
// RUN:   ToyView view(static_cast<std::uint8_t *>(rawState));
// RUN:   view.internal.scratch = static_cast<std::uint8_t>(view.a ^ view.b);
// RUN:   view.sum = static_cast<std::uint8_t>(view.a + view.b);
// RUN: }
// RUN:
// RUN: int main() {
// RUN:   constexpr std::size_t stride = ToyLayout::numStateBytes;
// RUN:   std::vector<std::uint8_t> states(stride * 4, 0);
// RUN:   arcilator::IndependentSweepConfig config{4, 1, ToyLayout::numStateBytes,
// RUN:                                             stride, states.size()};
// RUN:
// RUN:   auto fill = [](std::size_t stateIndex, std::uint8_t *state,
// RUN:                  std::size_t) {
// RUN:     Toy_initial(state);
// RUN:     ToyView view(state);
// RUN:     view.a = static_cast<std::uint8_t>(stateIndex);
// RUN:     view.b = static_cast<std::uint8_t>(20 + stateIndex);
// RUN:     view.sum = 0xff;
// RUN:   };
// RUN:
// RUN:   auto eval = [](std::uint8_t *state, std::size_t) {
// RUN:     Toy_eval(state);
// RUN:   };
// RUN:
// RUN:   auto compare = [](std::size_t stateIndex, const std::uint8_t *state,
// RUN:                     std::size_t) {
// RUN:     ToyView view(const_cast<std::uint8_t *>(state));
// RUN:     const auto expected =
// RUN:         static_cast<std::uint8_t>(stateIndex + 20 + stateIndex);
// RUN:     return view.sum == expected &&
// RUN:            view.internal.scratch ==
// RUN:                static_cast<std::uint8_t>(view.a ^ view.b);
// RUN:   };
// RUN:
// RUN:   const auto result =
// RUN:       arcilator::runIndependentSweep(states.data(), config, fill, eval,
// RUN:                                      compare);
// RUN:   if (result.caseCount != 4 || result.passedCount != 4 ||
// RUN:       result.failedCount != 0 || result.hasFailure)
// RUN:     return 1;
// RUN:
// RUN:   Toy single;
// RUN:   single.view.a = 7;
// RUN:   single.view.b = 9;
// RUN:   single.eval();
// RUN:   return single.view.sum == 16 ? 0 : 2;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator -I %t.dir %t.dir/test.cpp -o %t.dir/test
// RUN: %t.dir/test
