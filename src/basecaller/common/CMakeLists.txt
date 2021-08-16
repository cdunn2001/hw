
cmake_minimum_required(VERSION 3.20)

add_subdirectory(cuda)
add_subdirectory(simd)
add_subdirectory(test)

add_library(libMongoCommon STATIC
    AlignedCircularBuffer.h
    AlignedVector.h
    AutocorrAccumulator.h
    AutocorrAccumulator.cpp
    AutocorrAccumState.h
    AutocorrAccumState.cpp
    LaneArray.h
    LaneArray.cpp
    MongoConstants.h
    NumericUtil.h
    StatAccumulator.h
    StatAccumState.h
    # I think these should move to pa-common once tbb compilation
    # issues are resolved
    graphs/NodeMonitor.h
    graphs/NodeMonitor.cpp
    graphs/GraphsFwd.h
    graphs/GraphManager.h
    graphs/GraphManager.cpp
    graphs/GraphNodeBody.h
    graphs/GraphNodeBody.cpp
    utility/Overload.h
    utility/Overload.cpp
    utility/StrongTypedef.h
)

target_include_directories(libMongoCommon
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libMongoCommon
    PUBLIC
        libMongoCommonSimd
        libPacbioCplusplusApi
        libBoost
        libTbb
)
