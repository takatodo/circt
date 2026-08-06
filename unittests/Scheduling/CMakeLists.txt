set(SCHEDULING_TEST_SOURCES
  ResourceScheduleCertificateTest.cpp
)
if(ortools_FOUND)
  list(APPEND SCHEDULING_TEST_SOURCES
    CPSATResourceScheduleCertificateTest.cpp)
endif()

add_circt_unittest(CIRCTSchedulingTests
  ${SCHEDULING_TEST_SOURCES}
)

target_link_libraries(CIRCTSchedulingTests
  PRIVATE
  CIRCTScheduling
  MLIRIR
)
