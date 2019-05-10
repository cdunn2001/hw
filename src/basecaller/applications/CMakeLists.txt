
cmake_minimum_required(VERSION 3.8)

add_executable(mongo-basecaller-console-app
    ConsoleApp.cpp

)

target_link_libraries(mongo-basecaller-console-app
    PUBLIC
        libMongoBasecallerAnalyzer
        libProtoCommon
)