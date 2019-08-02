
cmake_minimum_required(VERSION 3.13)

include(pacbio-cplusplus-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoBasecallerTest OBJECT
    TestBaseliner.cpp
    TestHFMetricsFilter.cpp
    TestTraceAnalyzerTbb.cpp
    TestTraceHistogramAccumHost.cpp
    TestPulseAccumulator.cpp
)

target_link_libraries(libMongoBasecallerTest
    libMongoBasecallerAnalyzer
    libProtoCommon
)

# Define the test executable specific to mongo basecaller analyzer.
set(exec testMongoBasecallerTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoBasecallerTest
    gtest_main
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
