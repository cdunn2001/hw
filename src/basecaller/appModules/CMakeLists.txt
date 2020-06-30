
cmake_minimum_required(VERSION 3.8)

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
        BazWriter.h
        BazWriter.cpp
        BlockRepacker.h
        BlockRepacker.cpp
        TrivialRepacker.h
        TrivialRepacker.cpp
        TraceSaver.h
        TraceSaver.cpp
        TraceFileDataSource.h
        TraceFileDataSource.cpp
)

target_include_directories(libAppModules
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libAppModules
    PUBLIC
        libMongoBasecallerAnalyzer
        libProtoCommon
        libBazio
)

add_subdirectory(test)
