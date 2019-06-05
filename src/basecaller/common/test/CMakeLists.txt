
cmake_minimum_required(VERSION 3.13)

include(pacbio-cplusplus-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoCommonTest OBJECT
    TestLaneArray.cpp
)

target_link_libraries(libMongoCommonTest
    libMongoCommon
)

# Define the test executable specific to mongo basecaller analyzer.
add_executable(testMongoCommonTest
)

target_link_libraries(testMongoCommonTest
    libMongoCommonTest
    gtest_main
)