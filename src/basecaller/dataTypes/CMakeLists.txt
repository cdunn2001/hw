
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    configs/BasecallerAlgorithmConfig.cpp
    configs/BasecallerAlgorithmConfig.h
    configs/BasecallerBaselinerConfig.cpp
    configs/BasecallerBaselinerConfig.h
    configs/BasecallerDmeConfig.cpp
    configs/BasecallerDmeConfig.h
    configs/BasecallerFrameLabelerConfig.cpp
    configs/BasecallerFrameLabelerConfig.h
    configs/BasecallerMetricsConfig.cpp
    configs/BasecallerMetricsConfig.h
    configs/BasecallerPulseAccumConfig.cpp
    configs/BasecallerPulseAccumConfig.h
    configs/BasecallerPulseToBaseConfig.cpp
    configs/BasecallerPulseToBaseConfig.h
    configs/BasecallerTraceHistogramConfig.cpp
    configs/BasecallerTraceHistogramConfig.h
    configs/BasecallerSignalRangeEstimatorConfig.cpp
    configs/BasecallerSignalRangeEstimatorConfig.h
    configs/BatchLayoutConfig.h
    configs/BatchLayoutConfig.cpp
    configs/ConfigForward.h
    configs/MovieConfig.cpp
    configs/MovieConfig.h
    configs/SimulatedFaults.h
    configs/SmrtBasecallerConfig.cpp
    configs/SmrtBasecallerConfig.h
    configs/SourceConfig.h
    configs/StaticDetModelConfig.cpp
    configs/StaticDetModelConfig.h
    configs/SystemsConfig.h
    configs/SystemsConfig.cpp
    AnalogMode.h
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
    DetectionModelHost.h
    DetectionModelHost.cpp
    LaneDetectionModel.h
    LaneDetectionModel.cpp
    LabelsBatch.h
    LabelsBatch.cpp
    PoolHistogram.h
    PoolHistogram.cpp
    Pulse.h
    Pulse.cpp
    PulseBatch.h
    PulseBatch.cpp
    TraceAnalysisMetrics.h
    TraceBatch.h
    TraceBatch.cpp
    TrainedCartParams.h
    TrainedCartParams.cpp
    TrainedCartDevice.h
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
