
cmake_minimum_required(VERSION 3.8)

add_library(libMongoBasecallerAnalyzer STATIC
    ITraceAnalyzer.h
    ITraceAnalyzerAsync.h
    BatchAnalyzer.h
    BatchAnalyzer.cpp
)

target_link_libraries(libMongoBasecallerAnalyzer
    PUBLIC
        libMongoDataTypes
)
