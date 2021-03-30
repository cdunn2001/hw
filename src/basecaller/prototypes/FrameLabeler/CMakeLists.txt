project(CudaTemplate LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)

add_library(libFrameLabeler STATIC
    FrameLabeler.h
    FrameLabeler.cu
    FrameLabelerKernels.cuh
    FrameLabelerKernels.cu
    SubframeScorer.cu
    SubframeScorer.cuh
)

target_include_directories(libFrameLabeler
    PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR})
        
target_link_libraries(libFrameLabeler
    libProtoCommon)

set_target_properties(libFrameLabeler PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(libFrameLabeler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(FrameLabeler
    main.cpp
)

target_link_libraries(FrameLabeler
    libFrameLabeler)

set_target_properties(FrameLabeler PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(FrameLabeler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include(pacbio-gtest)

set(exec FrameLabelerTests)

add_executable(${exec}
    test/testFrameLabeler.cu
)

target_link_libraries(FrameLabelerTests
    libFrameLabeler
    libGtest
    libAppModules
)

target_include_directories(${exec}
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(${exec}
    libGtestWithMain)

add_test(${exec} ${exec} --gtest_output=xml:${exec}.xml)

set_target_properties(FrameLabelerTests PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(FrameLabelerTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

