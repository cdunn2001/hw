
cmake_minimum_required(VERSION 3.13)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoDataTypesTest OBJECT
    TestLabelsBatch.cpp
)

target_link_libraries(libMongoDataTypesTest
    libMongoDataTypes
    libGtest
)

# Define the test executable specific to mongo basecaller analyzer.
set(exec testMongoDataTypesTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoDataTypesTest
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
