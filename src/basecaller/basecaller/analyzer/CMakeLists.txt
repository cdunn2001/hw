
cmake_minimum_required(VERSION 3.20)

include(pacbio-tbb)

add_library(libMongoBasecallerAnalyzer STATIC
    AlgoFactory.h
    AlgoFactory.cpp
    BatchAnalyzer.h
    BatchAnalyzer.cpp
)

target_link_libraries(libMongoBasecallerAnalyzer
    PUBLIC
        libMongoDataTypes
    PRIVATE
        libMongoBasecallerTraceAnalysis
        libTbb
        libPacbioSmrtdata
        libBaselineFilter
)

target_include_directories(libMongoBasecallerAnalyzer
    PUBLIC
        ${MONGO_PROJECT_DIR}
)
