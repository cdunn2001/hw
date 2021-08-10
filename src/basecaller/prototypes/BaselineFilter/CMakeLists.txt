cmake_minimum_required(VERSION 3.20)

project(CudaTemplate LANGUAGES CXX CUDA)

add_library(libBaselineFilter STATIC
    BaselineRunner.cu
    BaselineRunner.h
    BaselineFilter.cu
    BaselineFilter.cuh
    BaselineFilterKernels.cu
    BaselineFilterKernels.cuh
    BlockCircularBuffer.cu
    BlockCircularBuffer.cuh
    CircularBufferKernels.cu
    CircularBufferKernels.cuh
    ExtremaFilter.cuh
    ExtremaFilter.cu
    ExtremaFilterKernels.cuh
    ExtremaFilterKernels.cu
    LocalCircularBuffer.cu
    LocalCircularBuffer.cuh
)

target_include_directories(libBaselineFilter
    PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR})
        
target_link_libraries(libBaselineFilter
    libProtoCommon)

set_target_properties(libBaselineFilter PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(libBaselineFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(BaselineFilter
    main.cpp
)

target_link_libraries(BaselineFilter
    libBaselineFilter)

set_target_properties(BaselineFilter PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(BaselineFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(pacbio-gtest)

add_executable(BaselineTests
    test/testExtremaFilter.cu
    test/testBaselineFilter.cu
    test/testCircularBuffer.cu
)

target_link_libraries(BaselineTests
    libProtoCommon
    libGtest
)

target_include_directories(BaselineTests
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(BaselineTests
    libGtestWithMain)

set_target_properties(BaselineTests PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(BaselineTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
