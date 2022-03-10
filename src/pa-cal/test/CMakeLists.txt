cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)
include(pacbio-eigen)

add_library(libTestPaCal OBJECT
    testPaCalProcess.cpp
    testPaCalSimulator.cpp
)

include_directories(..)

target_link_libraries(libTestPaCal
    PUBLIC
    lib-pa-cal
    libEigen
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
