
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    AnalogMode.h
    BasecallBatch.h
    BasecallBatch.cpp
    BasecallerConfig.h
    BaselinerStatAccumulator.h
    BaselinerStatAccumulator.cpp
    BaselineStats.h
    BaselineStats.cpp
    BasicTypes.h
    BatchData.h
    BatchData.cpp
    BatchMetadata.h
    CameraTraceBatch.h
    CameraTraceBatch.cpp
    ConfigForward.h
    DetectionModel.h
    DetectionModel.cpp
    DetectionModel.cuh
    DetectionModel.cu
    LaneDetectionModel.h
    LaneDetectionModel.cpp
    LabelsBatch.h
    LabelsBatch.cpp
    MovieConfig.h
    MovieConfig.cpp
    PoolHistogram.h
    PoolHistogram.cpp
    PrimaryConfig.h
    TraceBatch.h
    TraceBatch.cpp
)

set(CUDA_LIST
    BatchData.cuh
    BatchData.cu
)

add_library(libMongoDataTypes STATIC
    ${CPP_LIST}
    ${CUDA_LIST}
)

target_link_libraries(libMongoDataTypes
    PUBLIC
libBazio
        libMongoCommon
        libMongoCommonCuda
        libPacbioCplusplusApi
)

target_include_directories(libMongoDataTypes
    PUBLIC
        ${MONGO_PROJECT_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

 set_target_properties(libMongoDataTypes PROPERTIES
     CUDA_SEPARABLE_COMPILATION ON
 )
