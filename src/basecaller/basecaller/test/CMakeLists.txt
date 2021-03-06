
cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)
include(pacbio-gbench)

# Define "object library" to support top-level unit test target.
add_library(libMongoBasecallerTest OBJECT
    SpeedTestToggle.cpp
    TestBaseliner.cpp
    TestDetectionModelHost.cpp
    TestDmeEm.cpp
    TestFrameLabeler.cpp
    TestHFMetricsFilter.cpp
    TestBaselineStatsAggregator.cpp
    TestTraceHistogramAccumulator.cpp
    TestPulseAccumulator.cpp
)

target_link_libraries(libMongoBasecallerTest
    libMongoBasecallerAnalyzer
    libMongoBasecallerTraceAnalysis
    libProtoCommon
    libAppModules
    libTbb
    libGtest
)

# Define the test executable specific to mongo basecaller analyzer.
set(EXEC testMongoBasecallerTest)

add_executable(${EXEC}
    testMongoBasecallerTest.cpp
)

target_link_libraries(${EXEC}
    libMongoBasecallerTest
)

add_test(${EXEC} ${EXEC} --gtest_output=xml:${EXEC}.xml)

# Define benchmark project
add_executable(benchMongoBasecaller
    BenchBaseliner.cpp
)

target_link_libraries(benchMongoBasecaller
    libMongoBasecallerAnalyzer
    libProtoCommon
    libTbb
    libGbenchWithMain
)
