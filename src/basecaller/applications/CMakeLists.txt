
cmake_minimum_required(VERSION 3.8)

add_library(libSmrtBasecaller STATIC
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

target_include_directories(libSmrtBasecaller
    PUBLIC
        ${MONGO_PROJECT_DIR}
)

target_link_libraries(libSmrtBasecaller
    PUBLIC
        libMongoBasecallerAnalyzer
        libProtoCommon
        libBazio
)

add_executable(smrt-basecaller
        SmrtBasecaller.cpp
)


target_link_libraries(smrt-basecaller
    PRIVATE
        libSmrtBasecaller
        libPacbioHwMongo
)
add_subdirectory(test)
