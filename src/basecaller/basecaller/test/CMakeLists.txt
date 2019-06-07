
cmake_minimum_required(VERSION 3.13)

include(pacbio-cplusplus-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoBasecallerTest OBJECT
    TestBaseliner.cpp
    TestTraceAnalyzerTbb.cpp
)

target_link_libraries(libMongoBasecallerTest
    libMongoBasecallerAnalyzer
    libProtoCommon
)

# Define the test executable specific to mongo basecaller analyzer.
add_executable(testMongoBasecallerTest
)

target_link_libraries(testMongoBasecallerTest
    libMongoBasecallerTest
    gtest_main
)
