
cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoCommonTest OBJECT
    TestLaneArray.cpp
    TestGraphAPI.cpp
)

target_link_libraries(libMongoCommonTest
    libMongoCommon
    libGtest
)

# Define the test executable specific to mongo basecaller analyzer.
set(exec testMongoCommonTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoCommonTest
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
