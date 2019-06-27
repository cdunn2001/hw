
cmake_minimum_required(VERSION 3.13)

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
    DeviceMultiScaleBaseliner.h
    DeviceMultiScaleBaseliner.cu
    DeviceSGCFrameLabeler.h
    DeviceSGCFrameLabeler.cu
    HostMultiScaleBaseliner.h
    HostMultiScaleBaseliner.cpp
    HostNoOpBaseliner.h
    HostNoOpBaseliner.cpp
    HostSimulatedPulseAccumulator.h
    HostSimulatedPulseAccumulator.cpp
    PulseAccumulator.h
    PulseAccumulator.cpp
    TraceAnalysisForward.h
    TraceFilters.h
    TraceFilters.cpp
    TraceHistogramAccumulator.h
    TraceHistogramAccumulator.cpp
    TraceHistogramAccumHost.h
    TraceHistogramAccumHost.cpp
    WindowBuffer.h
)

target_include_directories(libMongoBasecallerTraceAnalysis
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libMongoBasecallerTraceAnalysis
    PUBLIC
        libMongoDataTypes
)
