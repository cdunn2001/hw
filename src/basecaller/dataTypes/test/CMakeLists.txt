
cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoDataTypesTest OBJECT
    TestLabelsBatch.cpp
    TestMetrics.cpp
)

include(pacbio-half)

target_link_libraries(libMongoDataTypesTest
    libMongoDataTypes
    libHalf
    libGtest
)

# Define the test executable specific to the data types library
set(exec testMongoDataTypesTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoDataTypesTest
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
