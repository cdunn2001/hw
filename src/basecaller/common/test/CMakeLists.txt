
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
set(exec testMongoCommonTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoCommonTest
    gtest_main
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)