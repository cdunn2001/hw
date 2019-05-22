
cmake_minimum_required(VERSION 3.8)

add_executable(mongo-basecaller
    ConsoleApp.cpp

)

target_link_libraries(mongo-basecaller
    PUBLIC
        libMongoBasecallerAnalyzer
        libProtoCommon
        libBazio
)