
cmake_minimum_required(VERSION 3.13)

include(pacbio-cplusplus-gtest)

# Define "object library" to support top-level unit test target.
add_library(libMongoCommonCudaTest OBJECT
    TestCudaEvents.cu
    TestStreamMonitors.cu
)

target_link_libraries(libMongoCommonCudaTest
    libMongoCommon
    libMongoCommonCuda
)

# Define the test executable specific to cuda common
set(exec testMongoCommonCudaTest)

add_executable(${exec})

target_link_libraries(${exec}
    libMongoCommonCudaTest
    gtest_main
)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)

set_target_properties(${exec} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
