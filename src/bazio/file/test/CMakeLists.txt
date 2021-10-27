cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libBazioFileTest OBJECT
    TestRLE.cpp
    TestZmwInfo.cpp
)

target_link_libraries(libBazioFileTest
    libBazioFile
    libGtest
)

# Define the test executable specific to bazio encoding
set(exec testBazioFile)

add_executable(${exec})

target_link_libraries(${exec}
    libBazioFileTest
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)