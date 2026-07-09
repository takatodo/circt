// REQUIRES: python
//
// This is a lit-only external-runner smoke test for independent testcase state
// slots. It keeps the generated arcilator model resident in one process and
// drives a small stateful datapath with per-case deterministic inputs. It does
// not add or require a runtime API.
//
// RUN: mkdir -p %t.dir
// RUN: arcilator %s --no-runtime --no-generate-driver --state-file %t.dir/state.json -o %t.dir/mixed.ll
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/state.json > %t.dir/mixed.h
// RUN: llc -O3 --filetype=obj %t.dir/mixed.ll -o %t.dir/mixed.o
// RUN: cat > %t.dir/test.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <cstring>
// RUN: #include <iostream>
// RUN: #include <limits>
// RUN: #include <vector>
// RUN: #include "mixed.h"
// RUN:
// RUN: static std::uint8_t inputA(std::size_t caseIndex,
// RUN:                            std::size_t stepIndex) {
// RUN:   return static_cast<std::uint8_t>((caseIndex * 13 + stepIndex * 19 + 0x31) & 0xff);
// RUN: }
// RUN:
// RUN: static std::uint8_t inputB(std::size_t caseIndex,
// RUN:                            std::size_t stepIndex) {
// RUN:   return static_cast<std::uint8_t>((caseIndex * 7 + stepIndex * 23 + 0x4d) & 0xff);
// RUN: }
// RUN:
// RUN: static bool selectFor(std::size_t caseIndex, std::size_t stepIndex) {
// RUN:   return ((caseIndex + stepIndex) & 1) != 0;
// RUN: }
// RUN:
// RUN: static bool enableFor(std::size_t caseIndex, std::size_t stepIndex) {
// RUN:   return ((caseIndex * 3 + stepIndex) % 5) != 0;
// RUN: }
// RUN:
// RUN: static std::size_t stepsFor(std::size_t caseIndex) {
// RUN:   return 1 + (caseIndex % 7);
// RUN: }
// RUN:
// RUN: struct RefMixedDatapath {
// RUN:   std::uint8_t acc = 0x13;
// RUN:   std::uint8_t hist = 0x27;
// RUN:
// RUN:   std::uint8_t y(std::uint8_t a, std::uint8_t b, bool sel) const {
// RUN:     const auto sumA = static_cast<std::uint8_t>(acc + a);
// RUN:     const auto sumB = static_cast<std::uint8_t>(hist + b);
// RUN:     const auto dat = sel ? sumB : sumA;
// RUN:     return static_cast<std::uint8_t>(dat + acc + hist);
// RUN:   }
// RUN:
// RUN:   void cycle(bool en, std::uint8_t a, std::uint8_t b, bool sel) {
// RUN:     const auto sumA = static_cast<std::uint8_t>(acc + a);
// RUN:     const auto sumB = static_cast<std::uint8_t>(hist + b);
// RUN:     const auto dat = sel ? sumB : sumA;
// RUN:     const auto nextAccData = static_cast<std::uint8_t>(dat + hist);
// RUN:     const auto nextHistData = static_cast<std::uint8_t>(acc + b);
// RUN:     if (en) {
// RUN:       acc = nextAccData;
// RUN:       hist = nextHistData;
// RUN:     }
// RUN:   }
// RUN: };
// RUN:
// RUN: static void initSlot(std::uint8_t *slot) {
// RUN:   std::memset(slot, 0, MixedDatapathBenchLayout::numStateBytes);
// RUN:   MixedDatapathBench_initial(slot);
// RUN: }
// RUN:
// RUN: static void cycle(MixedDatapathBenchView &view) {
// RUN:   view.clock = 0;
// RUN:   MixedDatapathBench_eval(view.state);
// RUN:   view.clock = 1;
// RUN:   MixedDatapathBench_eval(view.state);
// RUN: }
// RUN:
// RUN: int main() {
// RUN:   constexpr std::size_t caseCount = 40;
// RUN:   constexpr std::size_t stride = MixedDatapathBenchLayout::numStateBytes;
// RUN:   std::vector<std::uint8_t> states(caseCount * stride, 0);
// RUN:   std::uint64_t checksum = 0;
// RUN:   std::size_t passedCount = 0;
// RUN:   std::size_t firstFail = std::numeric_limits<std::size_t>::max();
// RUN:   std::size_t stepChecks = 0;
// RUN:
// RUN:   for (std::size_t caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
// RUN:     auto *slot = states.data() + caseIndex * stride;
// RUN:     initSlot(slot);
// RUN:     MixedDatapathBenchView view(slot);
// RUN:     RefMixedDatapath ref;
// RUN:     bool casePassed = true;
// RUN:
// RUN:     view.clock = 0;
// RUN:     view.en = 1;
// RUN:     view.sel = selectFor(caseIndex, 0);
// RUN:     view.a = inputA(caseIndex, 0);
// RUN:     view.b = inputB(caseIndex, 0);
// RUN:     MixedDatapathBench_eval(view.state);
// RUN:     casePassed &= view.y == ref.y(view.a, view.b, view.sel);
// RUN:     casePassed &= view.tap == ref.hist;
// RUN:     ++stepChecks;
// RUN:
// RUN:     const auto steps = stepsFor(caseIndex);
// RUN:     for (std::size_t stepIndex = 0; stepIndex < steps; ++stepIndex) {
// RUN:       const auto a = inputA(caseIndex, stepIndex);
// RUN:       const auto b = inputB(caseIndex, stepIndex);
// RUN:       const auto sel = selectFor(caseIndex, stepIndex);
// RUN:       const auto en = enableFor(caseIndex, stepIndex);
// RUN:       view.a = a;
// RUN:       view.b = b;
// RUN:       view.sel = sel;
// RUN:       view.en = en;
// RUN:       cycle(view);
// RUN:       ref.cycle(en, a, b, sel);
// RUN:       casePassed &= view.y == ref.y(a, b, sel);
// RUN:       casePassed &= view.tap == ref.hist;
// RUN:       ++stepChecks;
// RUN:     }
// RUN:
// RUN:     if (!casePassed && firstFail == std::numeric_limits<std::size_t>::max())
// RUN:       firstFail = caseIndex;
// RUN:   }
// RUN:
// RUN:   for (std::size_t caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
// RUN:     MixedDatapathBenchView view(states.data() + caseIndex * stride);
// RUN:     if (firstFail == std::numeric_limits<std::size_t>::max())
// RUN:       ++passedCount;
// RUN:     checksum += view.y;
// RUN:     checksum += static_cast<std::uint64_t>(view.tap) << 1;
// RUN:   }
// RUN:
// RUN:   const auto failedCount = caseCount - passedCount;
// RUN:   if (failedCount != 0)
// RUN:     return 1;
// RUN:   if (stepChecks != 195)
// RUN:     return 2;
// RUN:   if (checksum != 16194)
// RUN:     return 3;
// RUN:
// RUN:   std::cout << "case_count=" << caseCount
// RUN:             << " passed_count=" << passedCount
// RUN:             << " failed_count=" << failedCount
// RUN:             << " first_fail=none"
// RUN:             << " step_checks=" << stepChecks
// RUN:             << " checksum=" << checksum << "\n";
// RUN:   return 0;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator -I %t.dir %t.dir/test.cpp %t.dir/mixed.o -o %t.dir/test
// RUN: %t.dir/test | FileCheck --match-full-lines %s
//
// CHECK: case_count=40 passed_count=40 failed_count=0 first_fail=none step_checks=195 checksum=16194

module {
  hw.module @MixedDatapathBench(in %clock : i1, in %en : i1, in %sel : i1, in %a : i8, in %b : i8, out y : i8, out tap : i8) {
    %seq_clk = seq.to_clock %clock
    %acc = seq.firreg %nextAcc clock %seq_clk preset 0x13 : i8
    %hist = seq.firreg %nextHist clock %seq_clk preset 0x27 : i8
    %sumA = comb.add %acc, %a : i8
    %sumB = comb.add %hist, %b : i8
    %dat = comb.mux bin %sel, %sumB, %sumA : i8
    %nextAccData = comb.add %dat, %hist : i8
    %nextHistData = comb.add %acc, %b : i8
    %nextAcc = comb.mux bin %en, %nextAccData, %acc : i8
    %nextHist = comb.mux bin %en, %nextHistData, %hist : i8
    %y0 = comb.add %dat, %acc : i8
    %y = comb.add %y0, %hist : i8
    hw.output %y, %hist : i8, i8
  }
}
