// REQUIRES: python
//
// This is intentionally an external runner smoke test. It does not add an
// arcilator runtime API; it checks that the existing generated
// Layout/View/eval(void *) shape is enough to drive independent state slots.
//
// RUN: mkdir -p %t.dir
// RUN: arcilator %s --state-file %t.dir/state.json -o %t.dir/sweep_toy.ll
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/state.json > %t.dir/sweep_toy.h
// RUN: llc -O3 --filetype=obj %t.dir/sweep_toy.ll -o %t.dir/sweep_toy.o
// RUN: cat > %t.dir/test.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <cstring>
// RUN: #include <iostream>
// RUN: #include <vector>
// RUN: #include "sweep_toy.h"
// RUN:
// RUN: static void step(SweepToyView &view) {
// RUN:   view.clock = 0;
// RUN:   SweepToy_eval(view.state);
// RUN:   view.clock = 1;
// RUN:   SweepToy_eval(view.state);
// RUN: }
// RUN:
// RUN: int main() {
// RUN:   constexpr std::size_t stateCount = 32;
// RUN:   constexpr std::size_t steps = 3;
// RUN:   constexpr std::size_t stride = SweepToyLayout::numStateBytes;
// RUN:   std::vector<std::uint8_t> states(stateCount * stride, 0);
// RUN:   std::uint32_t bucketMask = 0;
// RUN:   std::uint64_t checksum = 0;
// RUN:   std::size_t passedCount = 0;
// RUN:
// RUN:   for (std::size_t i = 0; i < stateCount; ++i) {
// RUN:     auto *slot = states.data() + i * stride;
// RUN:     std::memset(slot, 0, stride);
// RUN:     SweepToyView view(slot);
// RUN:     view.inp = static_cast<std::uint16_t>(10 + i * 3);
// RUN:   }
// RUN:
// RUN:   for (std::size_t i = 0; i < stateCount; ++i) {
// RUN:     SweepToyView view(states.data() + i * stride);
// RUN:     for (std::size_t stepIndex = 0; stepIndex < steps; ++stepIndex)
// RUN:       step(view);
// RUN:   }
// RUN:
// RUN:   for (std::size_t i = 0; i < stateCount; ++i) {
// RUN:     SweepToyView view(states.data() + i * stride);
// RUN:     const auto inp = static_cast<std::uint16_t>(10 + i * 3);
// RUN:     const auto expected = static_cast<std::uint16_t>(2 * inp);
// RUN:     if (view.out != expected)
// RUN:       return 1;
// RUN:     ++passedCount;
// RUN:     bucketMask |= std::uint32_t{1} << (view.out & 31);
// RUN:     checksum += view.out;
// RUN:   }
// RUN:
// RUN:   if (passedCount != stateCount || bucketMask != 0x55555555u ||
// RUN:       checksum != 3616)
// RUN:     return 3;
// RUN:
// RUN:   std::cout << "case_count=" << stateCount
// RUN:             << " passed_count=" << passedCount
// RUN:             << " bucket_mask=0x" << std::hex << bucketMask
// RUN:             << " checksum=" << std::dec << checksum << "\n";
// RUN:
// RUN:   return 0;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator -I %t.dir %t.dir/test.cpp %t.dir/sweep_toy.o -o %t.dir/test
// RUN: %t.dir/test | FileCheck --match-full-lines %s
//
// CHECK: case_count=32 passed_count=32 bucket_mask=0x55555555 checksum=3616

hw.module @SweepToy(in %clock : !seq.clock, in %inp : i16, out out : i16) {
  %acc = seq.compreg %inp, %clock : i16
  %sum = comb.add %acc, %inp : i16
  hw.output %sum : i16
}
