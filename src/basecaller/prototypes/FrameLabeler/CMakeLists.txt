project(CudaTemplate LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.8)

add_executable(FrameLabeler
    main.cpp
    FrameLabeler.h
    FrameLabeler.cu
)

target_include_directories(FrameLabeler
    PUBLIC 
        ${CMAKE_CURRENT_LIST_DIR})
        
target_link_libraries(FrameLabeler
    libProtoCommon)

set_target_properties(FrameLabeler PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(FrameLabeler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)
