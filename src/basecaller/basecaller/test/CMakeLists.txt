
cmake_minimum_required(VERSION 3.13)

include(pacbio-cplusplus-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoBasecallerAnalyzerTest OBJECT
    TestTraceAnalyzerTbb.cpp
)

target_link_libraries(libMongoBasecallerAnalyzerTest
    libMongoBasecallerAnalyzer
)

# Define the test executable specific to mongo basecaller analyzer.
add_executable(testMongoBasecallerAnalyzerTest
)

target_link_libraries(testMongoBasecallerAnalyzerTest
    libMongoBasecallerAnalyzerTest
    gtest_main
)
