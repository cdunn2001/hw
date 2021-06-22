include(pacbio-gtest)

add_library(libTestPaCal OBJECT
    testPaCalProcess.cpp
)
    
target_link_libraries(libTestPaCal
    PUBLIC
    lib-pa-cal
    libPacbioCplusplusApi
    libGtestWithMain
)

add_executable(testPaCal
    testMain.cpp
    testMain.h
)

target_link_libraries(testPaCal
    libTestPaCal
)

add_test(testPaCal testPaCal --gtest_output=xml:testPaCal.xml)
