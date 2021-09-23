
cmake_minimum_required(VERSION 3.20)
include(pacbio-git-info)

add_executable(smrt-basecaller
        SmrtBasecaller.cpp
)

configure_file(numa_launch.sh numa_launch.sh COPYONLY)
configure_file(4x_launch.sh 4x_launch.sh COPYONLY)


target_link_libraries(smrt-basecaller
    PRIVATE
        libAppModules
        libPacbioHwAcquisition
        GitVersionInfo
)
add_subdirectory(test)
