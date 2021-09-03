cmake_minimum_required(VERSION 3.20)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libBazioEncodingTest OBJECT
    TestBazRoundTrip.cpp
    TestFieldSerializers.cpp
    TestFieldTransforms.cpp
    TestPulseToBaz.cu
)

target_link_libraries(libBazioEncodingTest
    libBazioEncoding
    # Not the most ideal dep.  It can be removed once DataParsing and BazEventData
    # get moved to the encoding directory, which can be done after the metrics
    # are moved to the new framework
    libBazio
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
