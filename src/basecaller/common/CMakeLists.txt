
cmake_minimum_required(VERSION 3.13)

add_subdirectory(cuda)
add_subdirectory(simd)

add_library(libMongoCommon STATIC
    AlignedCircularBuffer.h
    AutocorrAccumulator.h
    AutocorrAccumulator.cpp
    LaneArray.h
    LaneArray.cpp
    MongoConstants.h
    NumericUtil.h
    StatAccumulator.h
)

find_package(Boost)

target_include_directories(libMongoCommon
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libMongoCommon
    PUBLIC
        libMongoCommonSimd
        libPacbioCplusplusApi
        Boost::boost
        libTbb
)
