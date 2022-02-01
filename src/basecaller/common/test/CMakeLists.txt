
cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)
include(pacbio-gbench)
include(pacbio-eigen)

# Define "object library" to support top-level unit test target.
add_library(libMongoCommonTest OBJECT
    TestAutocorrAccumulator.cpp
    TestCircularArray.cpp
    TestLaneArray.cpp
    TestGraphAPI.cpp
    TestBatchDataSource.cpp
)

target_link_libraries(libMongoCommonTest
    libMongoCommon
    libMongoCommonCuda
    libEigen
    libGtest
)

# Define the test executable specific to mongo basecaller analyzer.
set(EXEC testMongoCommonTest)

add_executable(${EXEC})

target_link_libraries(${EXEC}
    libMongoCommonTest
    libGtestWithMain
)

add_test(${EXEC} ${EXEC} --gtest_output=xml:${EXEC}.xml)

# Define benchmark project
add_executable(benchMongoCommon BenchAutocorrAcc.cpp)

target_link_libraries(benchMongoCommon
    libMongoCommon
    libEigen
    libGbenchWithMain
)
