
cmake_minimum_required(VERSION 3.20)

add_subdirectory(test)

set(CPP_LIST
    configs/AnalysisConfig.cpp
    configs/AnalysisConfig.h
    configs/BasecallerAlgorithmConfig.cpp
    configs/BasecallerAlgorithmConfig.h
    configs/BasecallerBaselinerConfig.cpp
    configs/BasecallerBaselinerConfig.h
    configs/BasecallerBaselineStatsAggregatorConfig.cpp
    configs/BasecallerBaselineStatsAggregatorConfig.h
    configs/BasecallerDmeConfig.cpp
    configs/BasecallerDmeConfig.h
    configs/BasecallerFrameLabelerConfig.cpp
    configs/BasecallerFrameLabelerConfig.h
    configs/BasecallerMetricsConfig.cpp
    configs/BasecallerMetricsConfig.h
    configs/BasecallerPulseAccumConfig.cpp
    configs/BasecallerPulseAccumConfig.h
    configs/BasecallerTraceHistogramConfig.cpp
    configs/BasecallerTraceHistogramConfig.h
    configs/BatchLayoutConfig.h
    configs/BatchLayoutConfig.cpp
    configs/ConfigForward.h
    configs/PrelimHQConfig.cpp
    configs/PrelimHQConfig.h
    configs/RealTimeMetricsConfig.cpp
    configs/RealTimeMetricsConfig.h
    configs/SimulatedFaults.h
    configs/SmrtBasecallerConfig.cpp
    configs/SmrtBasecallerConfig.h
    configs/SourceConfig.h
    configs/StaticDetModelConfig.cpp
    configs/StaticDetModelConfig.h
    configs/SystemsConfig.h
    configs/SystemsConfig.cpp
    configs/TraceSaverConfig.h
    configs/TraceSaverConfig.cpp
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
    LaneDetectionModel.h
    LaneDetectionModel.cpp
    LabelsBatch.h
    LabelsBatch.cpp
    Metrics.h
    Metrics.cpp
    PoolHistogram.h
    PoolHistogram.cpp
    Pulse.h
    Pulse.cpp
    PulseBatch.h
    PulseBatch.cpp
    PulseFieldAccessors.h
    PulseFieldAccessors.cpp
    PulseGroups.h
    PulseGroups.cpp
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

include(pacbio-half)

target_link_libraries(libMongoDataTypes
    PUBLIC
libBazio
        libMongoCommon
        libMongoCommonCuda
        libPacbioCplusplusApi
        libPacbioHwAcquisition
        libHalf
)

target_include_directories(libMongoDataTypes
    PUBLIC
        ${MONGO_PROJECT_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

 set_target_properties(libMongoDataTypes PROPERTIES
     CUDA_SEPARABLE_COMPILATION ON
 )
