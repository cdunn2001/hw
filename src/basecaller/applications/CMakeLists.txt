
cmake_minimum_required(VERSION 3.8)

add_executable(mongo-basecaller
    ConsoleApp.cpp
)


add_library(libSmrtBasecaller STATIC
    Basecaller.h
    Basecaller.cpp
    BazWriter.h
    BazWriter.cpp
    CudaAllocator.h
    CudaAllocator.cpp
    Repacker.h
    Repacker.cpp
    TraceSaver.h
    TraceSaver.cpp
    TraceDataSource.h
    TraceDataSource.cpp
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

target_link_libraries(mongo-basecaller
    PUBLIC
        libMongoBasecallerAnalyzer
        libProtoCommon
        libBazio
        libTbb
)

target_link_libraries(smrt-basecaller
    PUBLIC
        libSmrtBasecaller
)
add_subdirectory(test)
