
cmake_minimum_required(VERSION 3.20)

# Intel complains in this file because there are some functions
# defined in an anonymous namespace that are never technically
# used.  However they are general things like move assignement
# operators that really should be defined, so that they are
# present if the code mutates slightly and needs them.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    SET_SOURCE_FILES_PROPERTIES(BlockRepacker.cpp PROPERTIES COMPILE_FLAGS -wd177)
endif()

add_library(libAppModules STATIC
        Basecaller.h
        Basecaller.cpp
        BazWriterBody.h
        BazWriterBody.cpp
        BlockRepacker.h
        BlockRepacker.cpp
        PrelimHQFilter.h
        PrelimHQFilter.cpp
        RealTimeMetrics.h
        RealTimeMetrics.cpp
        Repacker.h
        Repacker.cpp
        SmrtBasecallerProgress.h
        TrivialRepacker.h
        TrivialRepacker.cpp
        TraceSaver.h
        TraceSaver.cpp
        TraceFileDataSource.h
        TraceFileDataSource.cpp
        SimulatedDataSource.h
        SimulatedDataSource.cpp
)

target_link_libraries(libAppModules
    PUBLIC
        libMongoBasecallerAnalyzer
        libProtoCommon
        libBazio
)

add_subdirectory(test)
