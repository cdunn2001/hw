
cmake_minimum_required(VERSION 3.8)

add_library(libMongoBasecallerAnalyzer STATIC
    BatchAnalyzer.h
    BatchAnalyzer.cpp
    ITraceAnalyzer.h
    ITraceAnalyzerAsync.h
    TraceAnalyzerTbb.h
    TraceAnalyzerTbb.cpp
)

target_link_libraries(libMongoBasecallerAnalyzer
    PUBLIC
        libMongoDataTypes
)
