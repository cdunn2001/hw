
cmake_minimum_required(VERSION 3.13)
include(pacbio-git-info)

create_git_rev_header()

add_executable(smrt-basecaller
        SmrtBasecaller.cpp
        git-rev-pseudo-header.h
)

configure_file(numa_launch.sh numa_launch.sh COPYONLY)


target_link_libraries(smrt-basecaller
    PRIVATE
        libAppModules
        libPacbioHwMongo
)
add_subdirectory(test)
