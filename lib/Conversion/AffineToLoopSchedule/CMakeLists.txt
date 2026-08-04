add_circt_conversion_library(CIRCTAffineToLoopSchedule
  AffineToLoopSchedule.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithDialect

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTLoopSchedule
  )

if(ortools_FOUND)
  target_compile_definitions(obj.CIRCTAffineToLoopSchedule
    PRIVATE SCHEDULING_OR_TOOLS)
endif()
