cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libBazioWritingTest OBJECT
    TestBazAggregator.cpp
    TestGrowableArray.cpp
    TestPacketBuffer.cpp
    TestPacketBufferManager.cpp
)

target_link_libraries(libBazioWritingTest
    libBazioWriting
    libGtest
)

# Define the test executable specific to bazio writing
set(exec testBazioWriting)

add_executable(${exec})

target_link_libraries(${exec}
    libBazioWritingTest
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
