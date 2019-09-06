
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    AnalogMode.h
    BasecallerConfig.h
    BasecallingMetrics.h
    BasecallingMetrics.cpp
    BasecallingMetricsAccumulator.h
    BasecallingMetricsAccumulator.cpp
    BaselinerStatAccumulator.h
    BaselinerStatAccumulator.cpp
    BaselinerStatAccumState.h
    BasicTypes.h
    BatchData.h
    BatchData.cpp
    BatchMetadata.h
    BatchMetadata.cpp
    BatchMetrics.h
    BatchResult.h
    BatchResult.cpp
    BatchVectors.h
    BatchVectors.cpp
    BatchVectors.cuh
    BatchVectors.cu
    HQRFPhysicalStates.h
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
    StaticDetModelConfig.h
    StaticDetModelConfig.cpp
    TraceAnalysisMetrics.h
    TraceBatch.h
    TraceBatch.cpp
    TrainedCartParams.h
    TrainedCartParams.cpp
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
