set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  CPSATSchedulers.cpp
  LPSchedulers.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  Utilities.cpp
  )

set(SCHEDULING_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  Utilities.cpp
  )

set(SCHEDULING_LIBS
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  )

if(ortools_FOUND)
  list(APPEND SCHEDULING_SOURCES LPSchedulers.cpp CPSATSchedulers.cpp)
  list(APPEND SCHEDULING_LIBS ortools::ortools)
endif()

add_circt_library(CIRCTScheduling
  ${SCHEDULING_SOURCES}

  LINK_LIBS PUBLIC
  ${SCHEDULING_LIBS}
  )

# add_circt_library builds sources through an object library. Link OR-Tools to
# that target as well so its public compile definitions reach its sources.
if(ortools_FOUND)
  target_link_libraries(obj.CIRCTScheduling PRIVATE ortools::ortools)
endif()
