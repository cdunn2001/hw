
cmake_minimum_required(VERSION 3.13)

add_library(libMongoBasecallerTraceAnalysis STATIC
    Baseliner.h
    Baseliner.cpp
    TraceAnalysisForward.h
    TraceHistogramAccumulator.h
    TraceHistogramAccumulator.cpp
)

target_include_directories(libMongoBasecallerTraceAnalysis
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libMongoBasecallerTraceAnalysis
    PUBLIC
        libMongoDataTypes
)
