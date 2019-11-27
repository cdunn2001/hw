
cmake_minimum_required(VERSION 3.13)

include(pacbio-eigen)

add_library(libMongoBasecallerTraceAnalysis STATIC
    Baseliner.h
    Baseliner.cpp
    BaselinerParams.h
    BaselinerParams.cpp
    BlockFilterStage.h
    BlockFilterStage.cpp
    FrameLabeler.h
    FrameLabeler.cpp
    DetectionModelEstimator.h
    DetectionModelEstimator.cpp
    DeviceHFMetricsFilter.h
    DeviceHFMetricsFilter.cu
    DeviceMultiScaleBaseliner.h
    DeviceMultiScaleBaseliner.cu
    DevicePulseAccumulator.h
    DevicePulseAccumulator.cu
    DeviceSGCFrameLabeler.h
    DeviceSGCFrameLabeler.cu
    DmeDiagnostics.h
    DmeDiagnostics.cpp
    DmeEmHost.h
    DmeEmHost.cpp
    HFMetricsFilter.h
    HFMetricsFilter.cpp
    HostHFMetricsFilter.h
    HostHFMetricsFilter.cpp
    HostMultiScaleBaseliner.h
    HostMultiScaleBaseliner.cpp
    HostNoOpBaseliner.h
    HostNoOpBaseliner.cpp
    HostPulseAccumulator.cpp
    HostPulseAccumulator.h
    HostSimulatedPulseAccumulator.h
    HostSimulatedPulseAccumulator.cpp
    MaxLikelihoodDiagnostics.h
    PulseAccumulator.h
    PulseAccumulator.cpp
    SubframeLabelManager.h
    SubframeLabelManager.cpp
    TraceAnalysisForward.h
    TraceFilters.h
    TraceFilters.cpp
    TraceHistogramAccumHost.h
    TraceHistogramAccumHost.cpp
    TraceHistogramAccumulator.h
    TraceHistogramAccumulator.cpp
    WindowBuffer.h
)

target_include_directories(libMongoBasecallerTraceAnalysis
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libMongoBasecallerTraceAnalysis
    PUBLIC
        libMongoDataTypes
    PRIVATE
        libEigen
)
