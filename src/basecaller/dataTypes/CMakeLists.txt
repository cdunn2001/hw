
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    BasecallBatch.h
    BasecallBatch.cpp
    BatchMetadata.h
    TraceBatch.h
    TraceBatch.cpp
)

set(CUDA_LIST
    TraceBatch.cuh
    TraceBatch.cu
)

add_library(libMongoDataTypes STATIC
    ${CPP_LIST}
    # ${CUDA_LIST}
)

target_link_libraries(libMongoDataTypes
    PUBLIC
        libMongoCommonCudaMemory
        # libPacbioCplusplusApi
)

# set_target_properties(libCudaCommon PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# target_include_directories(libCudaCommon
#     PUBLIC 
#         "${CMAKE_CURRENT_LIST_DIR}/src"
#         ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# )
