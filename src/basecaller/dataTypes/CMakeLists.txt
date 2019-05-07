
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    AnalogMode.h
    BasecallBatch.h
    BasecallBatch.cpp
    BasecallerConfig.h
    BatchMetadata.h
    ConfigForward.h
    MovieConfig.h
    MovieConfig.cpp
    PrimaryConfig.h
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
        libMongoCommonCuda
        libPacbioCplusplusApi
)

target_include_directories(libMongoDataTypes
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)

# set_target_properties(libCudaCommon PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# target_include_directories(libCudaCommon
#     PUBLIC 
#         "${CMAKE_CURRENT_LIST_DIR}/src"
#         ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
# )
