
cmake_minimum_required(VERSION 3.13)

add_library(libMongoBasecallerTraceAnalysis STATIC
    Baseliner.h
    Baseliner.cpp
    BaselinerParams.h
    BaselinerParams.cpp
    FrameLabeler.h
    FrameLabeler.cpp
    DeviceMultiScaleBaseliner.h
    DeviceMultiScaleBaseliner.cu
    DeviceSGCFrameLabeler.h
    DeviceSGCFrameLabeler.cu
    DetectionModelEstimator.h
    DetectionModelEstimator.cpp
    BlockFilterStage.h
    BlockFilterStage.cpp
    DeviceMultiScaleBaseliner.h
    DeviceMultiScaleBaseliner.cu
    HostMultiScaleBaseliner.h
    HostMultiScaleBaseliner.cpp
    HostNoOpBaseliner.h
    HostNoOpBaseliner.cpp
    TraceAnalysisForward.h
    TraceFilters.h
    TraceFilters.cpp
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
)
