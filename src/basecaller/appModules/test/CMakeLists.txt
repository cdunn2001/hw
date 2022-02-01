
cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libTestAppModules OBJECT
    MockExperimentData.cpp
    TestBlockRepacker.cpp
    TestBasecaller.cpp
    TestSimulatedDataSource.cpp
    TestTraceFileDataSource.cpp
    TestTraceSaver.cpp
)

target_link_libraries(libTestAppModules
    libAppModules
    libGtest
)

# Define the test executable specific to the appModules
set(exec testAppModules)

add_executable(${exec})

target_link_libraries(${exec}
    libTestAppModules
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
