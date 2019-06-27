
cmake_minimum_required(VERSION 3.13)

add_subdirectory(cuda)
add_subdirectory(simd)
add_subdirectory(test)

add_library(libMongoCommon STATIC
    AlignedCircularBuffer.h
    AlignedVector.h
    AutocorrAccumulator.h
    AutocorrAccumulator.cpp
    AutocorrAccumState.h
    LaneArray.h
    LaneArray.cpp
    LaneArrayRef.h
    LaneMask.h
    MongoConstants.h
    NumericUtil.h
    StatAccumulator.h
    StatAccumState.h
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
