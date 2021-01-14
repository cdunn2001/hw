
cmake_minimum_required(VERSION 3.13)

add_executable(smrt-basecaller
        SmrtBasecaller.cpp
)

configure_file(numa_launch.sh numa_launch.sh COPYONLY)


target_link_libraries(smrt-basecaller
    PRIVATE
        libAppModules
        libPacbioHwMongo
)
add_subdirectory(test)
