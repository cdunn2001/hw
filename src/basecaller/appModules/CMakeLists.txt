
cmake_minimum_required(VERSION 3.8)

add_library(libAppModules STATIC
        Basecaller.h
        Basecaller.cpp
        BazWriter.h
        BazWriter.cpp
        Repacker.h
        Repacker.cpp
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

#add_subdirectory(test)
