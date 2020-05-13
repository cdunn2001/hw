
cmake_minimum_required(VERSION 3.13)

add_executable(smrt-basecaller
        SmrtBasecaller.cpp
)


target_link_libraries(smrt-basecaller
    PRIVATE
        libAppModules
        libPacbioHwMongo
)
add_subdirectory(test)
