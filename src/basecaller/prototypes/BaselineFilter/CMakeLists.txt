project(CudaTemplate LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)

add_executable(BaselineFilter
    main.cpp
    BaselineRunner.cu
    BaselineRunner.h
    BaselineFilter.cu
    BaselineFilter.cuh
    BaselineFilterKernels.cu
    BaselineFilterKernels.cuh
    ExtremaFilter.cuh
    ExtremaFilter.cu
    ExtremaFilterKernels.cuh
    ExtremaFilterKernels.cu
)

target_include_directories(BaselineFilter
    PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR})
        
target_link_libraries(BaselineFilter
    libProtoCommon)

set_target_properties(BaselineFilter PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(BaselineFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


include(pacbio-cplusplus-gtest)

add_executable(BaselineTests
    test/testExtremaFilter.cu
    test/testBaselineFilter.cu
)

target_link_libraries(BaselineTests
    libProtoCommon)

target_include_directories(BaselineTests
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(BaselineTests
    gtest_main)

set_target_properties(BaselineTests PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(BaselineTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
