cmake_minimum_required(VERSION 3.8)

set(SRC_LIST
    src/common/KernelThreadPool.h
    src/common/KernelThreadPool.cpp
    src/common/ZmwDataManager.h
    src/common/ZmwDataManager.cpp
    src/common/memory/UnifiedCudaArray.h
    src/common/memory/UnifiedCudaArray.cuh
    src/common/DataGenerators/BatchGenerator.h
    src/common/DataGenerators/GeneratorBase.h
    src/common/DataGenerators/GeneratorBase.cpp
    src/common/DataGenerators/TemplateGenerator.h
    src/common/DataGenerators/SawtoothGenerator.h
    src/common/DataGenerators/PicketFenceGenerator.h
    src/common/DataGenerators/PicketFenceGenerator.cpp
    src/common/DataGenerators/SignalGenerator.h
    src/common/DataGenerators/SignalGenerator.cpp
    src/common/DataGenerators/TraceFileReader.h
    src/common/DataGenerators/TraceFileReader.cpp
)

add_library(libProtoCommon STATIC
    ${SRC_LIST}
)

target_link_libraries(libProtoCommon 
    PUBLIC
        libMongoCommonCuda
        libSequelCommon1)
        
set_target_properties(libProtoCommon PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(libProtoCommon 
    PUBLIC 
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CURRENT_LIST_DIR}/src)

include(pacbio-gtest)

add_executable(CommonTests
        src/test/testSignalGenerator.cpp
)

target_link_libraries(CommonTests
    libProtoCommon
    gtest
)

target_include_directories(CommonTests
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(CommonTests
        gtest_main)

set_target_properties(CommonTests PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties(CommonTests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
