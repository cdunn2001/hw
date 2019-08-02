
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    AnalogMode.h
    BasecallBatch.h
    BasecallBatch.cpp
    BasecallerConfig.h
    BasecallingMetrics.h
    BasecallingMetrics.cpp
    BaselinerStatAccumulator.h
    BaselinerStatAccumulator.cpp
    BaselinerStatAccumState.h
    BaselineStats.h
    BaselineStats.cpp
    BasicTypes.h
    BatchData.h
    BatchData.cpp
    BatchMetadata.h
    BatchMetadata.cpp
    BatchResult.h
    BatchResult.cpp
    BatchVectors.h
    BatchVectors.cpp
    BatchVectors.cuh
    BatchVectors.cu
    CameraTraceBatch.h
    CameraTraceBatch.cpp
    ConfigForward.h
    DetectionModelHost.h
    DetectionModelHost.cpp
    LaneDetectionModel.h
    LaneDetectionModel.cpp
    LabelsBatch.h
    LabelsBatch.cpp
    MovieConfig.h
    MovieConfig.cpp
    PoolHistogram.h
    PoolHistogram.cpp
    PrimaryConfig.h
    Pulse.h
    Pulse.cpp
    PulseBatch.h
    PulseBatch.cpp
    TraceAnalysisMetrics.h
    TraceBatch.h
    TraceBatch.cpp
    UHistogramSimd.h
    UHistogramSimd.cpp
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
