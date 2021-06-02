# This exports 3 basic targets
#  libBazio - all library code and dependencies of that code

cmake_minimum_required (VERSION 3.13)

option(BUILD_SEQUEL_COMMON_TESTS "If ON, the unit and functional tests are built")
option(SHM_REALTIME              "If ON, build the pipeline using PoolShmRealtime instead of PoolSCIF (needed for Purley Skylake with no MICS)")

# Allow use of std::min in Visual Studio.
add_definitions(-DNOMINMAX)

include(pacbio-cplusplus-api)
include(pacbio-hdf5)
include(pacbio-tbb)
include(pacbio-boost)
include(pacbio-jsoncpp)
include(pacbio-half)

# libBazio - BAZ-related code

# BAZ file format version
set(Baz_MAJOR_VERSION 1)
set(Baz_MINOR_VERSION 6)
set(Baz_PATCH_VERSION 0)

# Baz IO version
# P4 version
if(NOT NO_P4VERSION)
    include(pacbio-versionheaderfile)
    get_perforce_change_number(P4VERSION)
endif()

# Generate header with versions
configure_file(template/BazVersion.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/BazVersion.h)

add_library(libBazio
    STATIC
        BazBuffer.h
        BazCore.h
        BazEventData.cpp
        BazEventData.h
        BazReader.cpp
        BazReader.h
        BazWriter.cpp
        BazWriter.h
        BlockActivityLabels.cpp
        BlockActivityLabels.h
        BlockLevelMetrics.cpp
        BlockLevelMetrics.h
        Codec.h
        DataParsing.cpp
        DataParsing.h
        FieldType.h
        FileFooter.h
        FileFooterBuilder.h
        FileHeader.cpp
        FileHeader.h
        FileHeaderBuilder.h
        FileHeaderBuilder.cpp
        FileHeaderValidator.h
        MetricBlock.h
        MetricData.h
        MetricField.h
        MetricFieldMap.cpp
        MetricFieldMap.h
        MetricFieldName.h
        MetricFrequency.h
        PacketField.h
        PacketFieldMap.cpp
        PacketFieldMap.h
        PacketFieldName.h
        PrimaryToBaz.cpp
        PrimaryToBaz.h
        RegionLabel.h
        RegionLabelType.h
        Sanity.h
        Simulation.cpp
        Simulation.h
        SimulationRNG.h
        SmartBazEnum.h
        SmartMemory.h
        SuperChunkMeta.h
        Timing.h
        ZmwSlice.h
        ZmwSliceHeader.h
)

set(BAZIO_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
target_include_directories(libBazio
    PUBLIC
#remove this dep
${BAZIO_PROJECT_DIR}/basecaller
        ${BAZIO_PROJECT_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/generated
)

include(pacbio-half)

target_link_libraries(libBazio
    PUBLIC
        libPacbioCplusplusApi
        libPacbioSmrtdata
        libJsoncpp
        libHalf
)


add_subdirectory(bazviewer)
