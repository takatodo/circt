// REQUIRES: python
//
// This is a resident external-runner smoke test. It does not add an arcilator
// runtime API; it documents the smallest useful runner shape for many
// independent testcase state slots in one process.
//
// RUN: mkdir -p %t.dir
// RUN: arcilator %s --no-runtime --no-generate-driver --state-file %t.dir/state.json -o %t.dir/shiftreg.ll
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/state.json > %t.dir/shiftreg.h
// RUN: llc -O3 --filetype=obj %t.dir/shiftreg.ll -o %t.dir/shiftreg.o
// RUN: cat > %t.dir/test.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <cstring>
// RUN: #include <iostream>
// RUN: #include <limits>
// RUN: #include <vector>
// RUN: #include "shiftreg.h"
// RUN:
// RUN: static std::uint8_t inputFor(std::size_t caseIndex,
// RUN:                              std::size_t stepIndex) {
// RUN:   return static_cast<std::uint8_t>((caseIndex * 17 + stepIndex * 29 + 0x5a) & 0xff);
// RUN: }
// RUN:
// RUN: static std::uint8_t expectedFor(std::size_t caseIndex,
// RUN:                                 std::size_t steps) {
// RUN:   if (steps == 0)
// RUN:     return 0xca;
// RUN:   if (steps == 1)
// RUN:     return 0x00;
// RUN:   if (steps == 2)
// RUN:     return 0xfe;
// RUN:   return inputFor(caseIndex, steps - 3);
// RUN: }
// RUN:
// RUN: static std::size_t stepsFor(std::size_t caseIndex) {
// RUN:   return caseIndex % 6;
// RUN: }
// RUN:
// RUN: static void cycle(ShiftRegBenchView &view) {
// RUN:   view.clock = 0;
// RUN:   ShiftRegBench_eval(view.state);
// RUN:   view.clock = 1;
// RUN:   ShiftRegBench_eval(view.state);
// RUN: }
// RUN:
// RUN: static void initSlot(std::uint8_t *slot) {
// RUN:   std::memset(slot, 0, ShiftRegBenchLayout::numStateBytes);
// RUN:   ShiftRegBench_initial(slot);
// RUN: }
// RUN:
// RUN: int main() {
// RUN:   constexpr std::size_t caseCount = 48;
// RUN:   constexpr std::size_t stride = ShiftRegBenchLayout::numStateBytes;
// RUN:   std::vector<std::uint8_t> states(caseCount * stride, 0);
// RUN:   std::uint64_t checksum = 0;
// RUN:   std::size_t passedCount = 0;
// RUN:   std::size_t firstFail = std::numeric_limits<std::size_t>::max();
// RUN:
// RUN:   for (std::size_t caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
// RUN:     auto *slot = states.data() + caseIndex * stride;
// RUN:     initSlot(slot);
// RUN:     ShiftRegBenchView view(slot);
// RUN:     view.en = 1;
// RUN:     view.clock = 0;
// RUN:     view.din = inputFor(caseIndex, 0);
// RUN:     ShiftRegBench_eval(view.state);
// RUN:     const auto steps = stepsFor(caseIndex);
// RUN:     for (std::size_t stepIndex = 0; stepIndex < steps; ++stepIndex) {
// RUN:       view.din = inputFor(caseIndex, stepIndex);
// RUN:       cycle(view);
// RUN:     }
// RUN:   }
// RUN:
// RUN:   for (std::size_t caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
// RUN:     ShiftRegBenchView view(states.data() + caseIndex * stride);
// RUN:     const auto steps = stepsFor(caseIndex);
// RUN:     const auto expected = expectedFor(caseIndex, steps);
// RUN:     if (view.dout != expected) {
// RUN:       if (firstFail == std::numeric_limits<std::size_t>::max())
// RUN:         firstFail = caseIndex;
// RUN:       continue;
// RUN:     }
// RUN:     ++passedCount;
// RUN:     checksum += view.dout;
// RUN:   }
// RUN:
// RUN:   const auto failedCount = caseCount - passedCount;
// RUN:   if (failedCount != 0)
// RUN:     return 1;
// RUN:   if (checksum != 6976)
// RUN:     return 2;
// RUN:
// RUN:   std::cout << "case_count=" << caseCount
// RUN:             << " passed_count=" << passedCount
// RUN:             << " failed_count=" << failedCount
// RUN:             << " first_fail=none"
// RUN:             << " checksum=" << checksum << "\n";
// RUN:   return 0;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator -I %t.dir %t.dir/test.cpp %t.dir/shiftreg.o -o %t.dir/test
// RUN: %t.dir/test | FileCheck --match-full-lines %s
//
// CHECK: case_count=48 passed_count=48 failed_count=0 first_fail=none checksum=6976

module {
  hw.module @ShiftRegBench(in %clock : i1, in %en : i1, in %din : i8, out dout : i8) {
    %seq_clk = seq.to_clock %clock
    %srA = seq.firreg %0 clock %seq_clk preset 0xFE : i8
    %srB = seq.firreg %1 clock %seq_clk : i8
    %srC = seq.firreg %2 clock %seq_clk preset 0xCA : i8
    %0 = comb.mux bin %en, %din, %srA : i8
    %1 = comb.mux bin %en, %srA, %srB : i8
    %2 = comb.mux bin %en, %srB, %srC : i8
    hw.output %srC : i8
  }
}
