
cmake_minimum_required(VERSION 3.8)

set(CPP_LIST
    TraceBatch.h
    TraceBatch.cpp
)

set(CUDA_LIST
    TraceBatch.cuh
    TraceBatch.cu
)

add_library(libMongoDataTypes
    ${CPP_LIST}
    ${CUDA_LIST}
)

# target_link_libraries(libCudaCommon 
#     PUBLIC 
#         libPacbioCplusplusApi
#         ${CMAKE_CUDA_LIBRARIES}
# )

# set_target_properties(libCudaCommon PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# target_include_directories(libCudaCommon
#     PUBLIC 
#         "${CMAKE_CURRENT_LIST_DIR}/src"
#         ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# )
