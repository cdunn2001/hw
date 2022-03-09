cmake_minimum_required (VERSION 3.20)

###############################################################################

# project config and version

include(pacbio-git-info)

include(pacbio-cmake-config-vars)

generate_config_header(ppa-config.h PpaApplicationConfig)


###############################################################################

add_library(libPpaApp STATIC
    ${CMAKE_CURRENT_BINARY_DIR}/config/ppa-config.h
)

target_link_libraries(libPpaApp
    PUBLIC
        libPpaBase
        libPpaInterfaces
        libSequelCommon1
)

###############################################################################

#  baz2bam

add_library(libbaz2bam STATIC
    ConvertBaz2Bam.cpp
    OomScoreMonitor.h
    OomScoreMonitor.cpp
    PpaAlgoConfig.cpp
    PpaProgressMessage.h
    PpaProgressMessage.cpp
)

target_link_libraries(libbaz2bam
    PUBLIC
        libPpaApp
        GitVersionInfo
)

add_executable(baz2bam
    baz2bam.cpp
)

target_link_libraries(baz2bam
    PUBLIC
        libbaz2bam
    PRIVATE
        SegFault
)

if (Baz2bam_Jemalloc)
    target_link_libraries(baz2bam PUBLIC libJemalloc -pthread)
endif()

###############################################################################

#  bam2bam

add_library(libbam2bam STATIC
    ConvertBam2Bam.cpp
    PpaAlgoConfig.cpp
)

target_link_libraries(libbam2bam
    PUBLIC
        libPpaApp
        GitVersionInfo
)

add_executable(bam2bam
    bam2bam.cpp
)

target_link_libraries(bam2bam
    PUBLIC
        libbam2bam
    PRIVATE
        SegFault
)

###############################################################################

#  recalladapters

add_executable(recalladapters
    recalladapters.cpp
)

target_link_libraries(recalladapters
    PUBLIC
        libbam2bam
    PRIVATE
        SegFault
)

###############################################################################

#  adapterlab

add_executable(adapterlab
    adapterlabeler.cpp
)

target_link_libraries(adapterlab
    PUBLIC
        libbam2bam
    PRIVATE
        SegFault
)

###############################################################################

# ppa-reducestats

include(pacbio-jsoncpp)

add_executable(ppa-reducestats
    ppa-reducestats.cpp
)

target_link_libraries(ppa-reducestats
    PUBLIC
        libSequelCommon1
        libJsoncpp
        GitVersionInfo
    PRIVATE
        SegFault
)

###############################################################################

#  bampaverify

add_executable(bampaverify
    bampaverify.cpp
)

target_link_libraries(bampaverify
    PUBLIC
        libPpaBase
        libPpaInterfaces
        GitVersionInfo
    PRIVATE
        SegFault
        pthread
)
