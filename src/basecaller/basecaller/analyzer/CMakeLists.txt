
cmake_minimum_required(VERSION 3.13)

include(pacbio-tbb)

add_library(libMongoBasecallerAnalyzer STATIC
    BatchAnalyzer.h
    BatchAnalyzer.cpp
    ITraceAnalyzer.h
    ITraceAnalyzer.cpp
    ITraceAnalyzerAsync.h
    TraceAnalyzerTbb.h
    TraceAnalyzerTbb.cpp
)

target_link_libraries(libMongoBasecallerAnalyzer
    PUBLIC
        libMongoDataTypes
    PRIVATE
        libTbb
)

target_include_directories(libMongoBasecallerAnalyzer
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
)
