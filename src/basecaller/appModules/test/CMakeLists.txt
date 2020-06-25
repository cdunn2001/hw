
cmake_minimum_required(VERSION 3.13)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libTestAppModules OBJECT
    TestBlockRepacker.cpp
    TestBasecaller.cpp
)

target_link_libraries(libTestAppModules
    libAppModules
    libGtest
)

# Define the test executable specific to mongo basecaller analyzer.
set(exec testAppModules)

add_executable(${exec})

target_link_libraries(${exec}
    libTestAppModules
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
