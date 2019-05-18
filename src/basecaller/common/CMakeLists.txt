
cmake_minimum_required(VERSION 3.13)

add_subdirectory(cuda)
add_subdirectory(simd)

add_library(libMongoCommon INTERFACE
#    NumericUtil.h
#    StatAccumulator.h
)

target_include_directories(libMongoCommon
    INTERFACE
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libMongoCommon
    INTERFACE
        libMongoCommonSimd
)
