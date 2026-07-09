// REQUIRES: python
//
// This is a lit-only RFC specimen for a reusable resident independent-state
// runner shape. It intentionally does not add an arcilator runtime API, CMake
// wiring, install hook, manifest format, thread pool, or GPU backend.
//
// RUN: split-file %s %t.dir
// RUN: arcilator %t.dir/shift.mlir --no-runtime --no-generate-driver --state-file %t.dir/shift.json -o %t.dir/shift.ll
// RUN: arcilator %t.dir/mixed.mlir --no-runtime --no-generate-driver --state-file %t.dir/mixed.json -o %t.dir/mixed.ll
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/shift.json > %t.dir/shift.h
// RUN: %PYTHON% %CIRCT_SOURCE%/tools/arcilator/arcilator-header-cpp.py %t.dir/mixed.json > %t.dir/mixed.h
// RUN: llc -O3 --filetype=obj %t.dir/shift.ll -o %t.dir/shift.o
// RUN: llc -O3 --filetype=obj %t.dir/mixed.ll -o %t.dir/mixed.o
// RUN: cat > %t.dir/test.cpp <<'EOF'
// RUN: #include <cstdint>
// RUN: #include <cstring>
// RUN: #include <iostream>
// RUN: #include <limits>
// RUN: #include <vector>
// RUN: #include "shift.h"
// RUN: #include "mixed.h"
// RUN:
// RUN: struct BatchResult {
// RUN:   std::size_t caseCount = 0;
// RUN:   std::size_t passedCount = 0;
// RUN:   std::size_t firstFail = std::numeric_limits<std::size_t>::max();
// RUN:   std::size_t stepChecks = 0;
// RUN:   std::uint64_t checksum = 0;
// RUN:
// RUN:   std::size_t failedCount() const { return caseCount - passedCount; }
// RUN: };
// RUN:
// RUN: template <typename Traits>
// RUN: struct ResidentBatchRunner {
// RUN:   static BatchResult run() {
// RUN:     BatchResult result;
// RUN:     result.caseCount = Traits::caseCount;
// RUN:     constexpr std::size_t stride = Traits::Layout::numStateBytes;
// RUN:     std::vector<std::uint8_t> states(result.caseCount * stride, 0);
// RUN:
// RUN:     for (std::size_t caseIndex = 0; caseIndex < result.caseCount; ++caseIndex) {
// RUN:       auto *slot = states.data() + caseIndex * stride;
// RUN:       std::memset(slot, 0, stride);
// RUN:       Traits::initial(slot);
// RUN:       typename Traits::View view(slot);
// RUN:       typename Traits::Reference ref;
// RUN:       bool passed = Traits::initialCheck(view, ref, caseIndex, result.stepChecks);
// RUN:
// RUN:       const auto steps = Traits::stepsFor(caseIndex);
// RUN:       for (std::size_t stepIndex = 0; stepIndex < steps; ++stepIndex)
// RUN:         passed &= Traits::stepAndCheck(view, ref, caseIndex, stepIndex,
// RUN:                                        result.stepChecks);
// RUN:
// RUN:       if (passed)
// RUN:         ++result.passedCount;
// RUN:       else if (result.firstFail == std::numeric_limits<std::size_t>::max())
// RUN:         result.firstFail = caseIndex;
// RUN:       result.checksum += Traits::checksum(view);
// RUN:     }
// RUN:
// RUN:     return result;
// RUN:   }
// RUN: };
// RUN:
// RUN: static void printResult(const char *model, const BatchResult &result) {
// RUN:   std::cout << "model=" << model
// RUN:             << " case_count=" << result.caseCount
// RUN:             << " passed_count=" << result.passedCount
// RUN:             << " failed_count=" << result.failedCount()
// RUN:             << " first_fail=";
// RUN:   if (result.firstFail == std::numeric_limits<std::size_t>::max())
// RUN:     std::cout << "none";
// RUN:   else
// RUN:     std::cout << result.firstFail;
// RUN:   std::cout << " step_checks=" << result.stepChecks
// RUN:             << " checksum=" << result.checksum << "\n";
// RUN: }
// RUN:
// RUN: static std::uint8_t shiftInputFor(std::size_t caseIndex,
// RUN:                                  std::size_t stepIndex) {
// RUN:   return static_cast<std::uint8_t>((caseIndex * 17 + stepIndex * 29 + 0x5a) & 0xff);
// RUN: }
// RUN:
// RUN: struct RefShiftReg {
// RUN:   std::uint8_t srA = 0xfe;
// RUN:   std::uint8_t srB = 0x00;
// RUN:   std::uint8_t srC = 0xca;
// RUN:
// RUN:   void cycle(bool en, std::uint8_t din) {
// RUN:     const auto nextA = en ? din : srA;
// RUN:     const auto nextB = en ? srA : srB;
// RUN:     const auto nextC = en ? srB : srC;
// RUN:     srA = nextA;
// RUN:     srB = nextB;
// RUN:     srC = nextC;
// RUN:   }
// RUN: };
// RUN:
// RUN: struct ShiftRegTraits {
// RUN:   using Layout = ShiftRegBenchLayout;
// RUN:   using View = ShiftRegBenchView;
// RUN:   using Reference = RefShiftReg;
// RUN:   static constexpr std::size_t caseCount = 48;
// RUN:
// RUN:   static void initial(std::uint8_t *slot) { ShiftRegBench_initial(slot); }
// RUN:
// RUN:   static std::size_t stepsFor(std::size_t caseIndex) {
// RUN:     return caseIndex % 6;
// RUN:   }
// RUN:
// RUN:   static void cycle(View &view) {
// RUN:     view.clock = 0;
// RUN:     ShiftRegBench_eval(view.state);
// RUN:     view.clock = 1;
// RUN:     ShiftRegBench_eval(view.state);
// RUN:   }
// RUN:
// RUN:   static bool initialCheck(View &view, Reference &ref,
// RUN:                            std::size_t caseIndex,
// RUN:                            std::size_t &stepChecks) {
// RUN:     view.en = 1;
// RUN:     view.clock = 0;
// RUN:     view.din = shiftInputFor(caseIndex, 0);
// RUN:     ShiftRegBench_eval(view.state);
// RUN:     ++stepChecks;
// RUN:     return view.dout == ref.srC;
// RUN:   }
// RUN:
// RUN:   static bool stepAndCheck(View &view, Reference &ref,
// RUN:                            std::size_t caseIndex,
// RUN:                            std::size_t stepIndex,
// RUN:                            std::size_t &stepChecks) {
// RUN:     const auto din = shiftInputFor(caseIndex, stepIndex);
// RUN:     view.din = din;
// RUN:     cycle(view);
// RUN:     ref.cycle(true, din);
// RUN:     ++stepChecks;
// RUN:     return view.dout == ref.srC;
// RUN:   }
// RUN:
// RUN:   static std::uint64_t checksum(const View &view) { return view.dout; }
// RUN: };
// RUN:
// RUN: static std::uint8_t mixedInputA(std::size_t caseIndex,
// RUN:                                  std::size_t stepIndex) {
// RUN:   return static_cast<std::uint8_t>((caseIndex * 13 + stepIndex * 19 + 0x31) & 0xff);
// RUN: }
// RUN:
// RUN: static std::uint8_t mixedInputB(std::size_t caseIndex,
// RUN:                                  std::size_t stepIndex) {
// RUN:   return static_cast<std::uint8_t>((caseIndex * 7 + stepIndex * 23 + 0x4d) & 0xff);
// RUN: }
// RUN:
// RUN: static bool mixedSelectFor(std::size_t caseIndex, std::size_t stepIndex) {
// RUN:   return ((caseIndex + stepIndex) & 1) != 0;
// RUN: }
// RUN:
// RUN: static bool mixedEnableFor(std::size_t caseIndex, std::size_t stepIndex) {
// RUN:   return ((caseIndex * 3 + stepIndex) % 5) != 0;
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
// RUN: struct MixedDatapathTraits {
// RUN:   using Layout = MixedDatapathBenchLayout;
// RUN:   using View = MixedDatapathBenchView;
// RUN:   using Reference = RefMixedDatapath;
// RUN:   static constexpr std::size_t caseCount = 40;
// RUN:
// RUN:   static void initial(std::uint8_t *slot) { MixedDatapathBench_initial(slot); }
// RUN:
// RUN:   static std::size_t stepsFor(std::size_t caseIndex) {
// RUN:     return 1 + (caseIndex % 7);
// RUN:   }
// RUN:
// RUN:   static void cycle(View &view) {
// RUN:     view.clock = 0;
// RUN:     MixedDatapathBench_eval(view.state);
// RUN:     view.clock = 1;
// RUN:     MixedDatapathBench_eval(view.state);
// RUN:   }
// RUN:
// RUN:   static bool check(const View &view, const Reference &ref,
// RUN:                     std::uint8_t a, std::uint8_t b, bool sel) {
// RUN:     return view.y == ref.y(a, b, sel) && view.tap == ref.hist;
// RUN:   }
// RUN:
// RUN:   static bool initialCheck(View &view, Reference &ref,
// RUN:                            std::size_t caseIndex,
// RUN:                            std::size_t &stepChecks) {
// RUN:     view.clock = 0;
// RUN:     view.en = 1;
// RUN:     view.sel = mixedSelectFor(caseIndex, 0);
// RUN:     view.a = mixedInputA(caseIndex, 0);
// RUN:     view.b = mixedInputB(caseIndex, 0);
// RUN:     MixedDatapathBench_eval(view.state);
// RUN:     ++stepChecks;
// RUN:     return check(view, ref, view.a, view.b, view.sel);
// RUN:   }
// RUN:
// RUN:   static bool stepAndCheck(View &view, Reference &ref,
// RUN:                            std::size_t caseIndex,
// RUN:                            std::size_t stepIndex,
// RUN:                            std::size_t &stepChecks) {
// RUN:     const auto a = mixedInputA(caseIndex, stepIndex);
// RUN:     const auto b = mixedInputB(caseIndex, stepIndex);
// RUN:     const auto sel = mixedSelectFor(caseIndex, stepIndex);
// RUN:     const auto en = mixedEnableFor(caseIndex, stepIndex);
// RUN:     view.a = a;
// RUN:     view.b = b;
// RUN:     view.sel = sel;
// RUN:     view.en = en;
// RUN:     cycle(view);
// RUN:     ref.cycle(en, a, b, sel);
// RUN:     ++stepChecks;
// RUN:     return check(view, ref, a, b, sel);
// RUN:   }
// RUN:
// RUN:   static std::uint64_t checksum(const View &view) {
// RUN:     return view.y + (static_cast<std::uint64_t>(view.tap) << 1);
// RUN:   }
// RUN: };
// RUN:
// RUN: int main() {
// RUN:   const auto shift = ResidentBatchRunner<ShiftRegTraits>::run();
// RUN:   const auto mixed = ResidentBatchRunner<MixedDatapathTraits>::run();
// RUN:   printResult("shiftreg", shift);
// RUN:   printResult("mixed", mixed);
// RUN:
// RUN:   const auto combinedCases = shift.caseCount + mixed.caseCount;
// RUN:   const auto combinedChecks = shift.stepChecks + mixed.stepChecks;
// RUN:   const auto combinedChecksum = shift.checksum + mixed.checksum;
// RUN:   if (shift.failedCount() != 0 || mixed.failedCount() != 0)
// RUN:     return 1;
// RUN:   if (shift.stepChecks != 168 || shift.checksum != 6976)
// RUN:     return 2;
// RUN:   if (mixed.stepChecks != 195 || mixed.checksum != 16194)
// RUN:     return 3;
// RUN:   if (combinedCases != 88 || combinedChecks != 363 ||
// RUN:       combinedChecksum != 23170)
// RUN:     return 4;
// RUN:
// RUN:   std::cout << "models=2 combined_cases=" << combinedCases
// RUN:             << " combined_step_checks=" << combinedChecks
// RUN:             << " combined_checksum=" << combinedChecksum << "\n";
// RUN:   return 0;
// RUN: }
// RUN: EOF
// RUN: %host_cxx -std=c++17 -I %CIRCT_SOURCE%/tools/arcilator -I %t.dir %t.dir/test.cpp %t.dir/shift.o %t.dir/mixed.o -o %t.dir/test
// RUN: %t.dir/test | FileCheck --match-full-lines %s
//
// CHECK: model=shiftreg case_count=48 passed_count=48 failed_count=0 first_fail=none step_checks=168 checksum=6976
// CHECK: model=mixed case_count=40 passed_count=40 failed_count=0 first_fail=none step_checks=195 checksum=16194
// CHECK: models=2 combined_cases=88 combined_step_checks=363 combined_checksum=23170

//--- shift.mlir
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

//--- mixed.mlir
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
