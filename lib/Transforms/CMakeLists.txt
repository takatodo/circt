add_circt_library(CIRCTTransforms
  AffineSlidingReuseSupport.cpp
  AffineSlidingWindowReuse.cpp
  CellIFT.cpp
  FlattenMemRefs.cpp
  HierarchicalRunner.cpp
  IndexSwitchToIf.cpp
  InsertMergeBlocks.cpp
  ConvertIndexToUInt.cpp
  MapArithToComb.cpp
  MaximizeSSA.cpp
  MemoryBanking.cpp
  PrintOpCount.cpp
  StripDebugInfoWithPred.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTOpCountAnalysis
  MLIRArithDialect
  MLIRAffineAnalysis
  MLIRControlFlowDialect
  MLIRControlFlowTransforms
  MLIRFuncDialect
  MLIRIR
  MLIRAnalysis
  MLIRMemRefDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineDialect
  MLIRSCFDialect

  DEPENDS
  CIRCTTransformsPassIncGen
  MLIRArithDialect
)
