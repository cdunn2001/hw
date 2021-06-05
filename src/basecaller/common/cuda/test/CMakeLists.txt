
cmake_minimum_required(VERSION 3.13)

include(pacbio-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoCommonCudaTest OBJECT
    TestCudaEvents.cu
    TestCudaTuple.cu
    TestCudaArray.cu
    TestDeviceAllocationStash.cu
    TestManagedAllocations.cpp
    TestPBCudaSimd.cpp
    TestStreamMonitors.cu
)

target_link_libraries(libMongoCommonCudaTest
    libMongoCommon
    libMongoCommonCuda
    libGtest
)

# Define the test executable specific to cuda common
set(exec testMongoCommonCudaTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoCommonCudaTest
    libGtestWithMain
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)

set_target_properties(${exec} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
