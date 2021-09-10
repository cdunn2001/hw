cmake_minimum_required(VERSION 3.20)

project(CudaTemplate LANGUAGES CXX CUDA)

add_library(libBaselineFilter STATIC
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
    BaselineRunner.cu
    BaselineRunner.h
)

target_link_libraries(BaselineFilter
    libBaselineFilter
    libMongoBasecallerTraceAnalysis 
)

set_target_properties(BaselineFilter PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(BaselineFilter PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(pacbio-gtest)

set(testExec BaselineTests)

add_executable(${testExec}
    test/testExtremaFilter.cu
    test/testBaselineFilter.cu
    test/testCircularBuffer.cu
)

target_link_libraries(${testExec}
    libProtoCommon
    libGtest
    libMongoBasecallerTraceAnalysis 
    libAppModules
)

target_include_directories(${testExec}
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(${testExec}
    libGtestWithMain)

add_test(${testExec} ${testExec} -- gtest_output=xml:${testExec}.xml)

set_target_properties(BaselineTests PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(BaselineTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
