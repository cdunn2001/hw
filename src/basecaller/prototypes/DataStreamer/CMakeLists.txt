project(CudaTemplate LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)

add_executable(DataStreamer
    main.cpp
    StreamingTestRunner.h
    StreamingTestRunner.cu
)

target_include_directories(libProtoCommon 
    PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR})
        
target_link_libraries(DataStreamer 
    libProtoCommon)

set_target_properties(DataStreamer PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(DataStreamer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

