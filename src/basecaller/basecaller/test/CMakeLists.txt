
cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoBasecallerTest OBJECT
    SpeedTestToggle.cpp
    TestHostBaseliner.cpp
    TestDeviceBaseliner.cpp
    TestDmeEm.cpp
    TestHFMetricsFilter.cpp
    TestBaselineStatsAggregator.cpp
    TestTraceHistogramAccumulator.cpp
    TestPulseAccumulator.cpp
)

target_link_libraries(libMongoBasecallerTest
    libMongoBasecallerAnalyzer
    libProtoCommon
    libTbb
    libGtest
    libAppModules
)

# Define the test executable specific  the core basecaller analyzer.
set(exec testMongoBasecallerTest)

add_executable(${exec}
    testMongoBasecallerTest.cpp
)

target_link_libraries(${exec}
    libMongoBasecallerTest
    libGtest
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
