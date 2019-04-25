
cmake_minimum_required(VERSION 3.8)

add_subdirectory(memory)

set(SRC_LIST
    CudaFunctionDecorators.h
    DisableBoostOnDevice.h
    PBCudaRuntime.h
    PBCudaRuntime.cpp
)

add_library(libMongoCommonCuda STATIC
   ${SRC_LIST}
)

# target_link_libraries(libMongoCommonCuda 
#     PUBLIC 
#         libPacbioCplusplusApi
#         ${CMAKE_CUDA_LIBRARIES})

# set_target_properties(libMongoCommonCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(libMongoCommonCuda
#     PUBLIC 
#         "${CMAKE_CURRENT_LIST_DIR}/src"
#         ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

