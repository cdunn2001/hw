cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libBazioEncodingTest OBJECT
    TestBazToPulse.cpp
    TestFieldSerializers.cpp
    TestFieldTransforms.cpp
    TestPulseToBaz.cu
)

target_link_libraries(libBazioEncodingTest
    libBazioEncoding
    libGtest
)

# Define the test executable specific to bazio encoding
set(exec testBazioEncoding)

add_executable(${exec})

target_link_libraries(${exec}
    libBazioEncodingTest
    libMongoCommonCuda
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)
