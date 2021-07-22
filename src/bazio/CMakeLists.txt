# This exports 3 basic targets
#  libBazio - all library code and dependencies of that code

cmake_minimum_required (VERSION 3.13)

# Allow use of std::min in Visual Studio.
add_definitions(-DNOMINMAX)

include(pacbio-cplusplus-api)
include(pacbio-boost)
include(pacbio-jsoncpp)
include(pacbio-half)

# libBazio - BAZ-related code

# BAZ file format version
set(Baz_MAJOR_VERSION 2)
set(Baz_MINOR_VERSION 0)
set(Baz_PATCH_VERSION 0)

# Explicitly set BazIO_VERSION
set(BazIO_VERSION "0.0.0")

# Explicitly set BAZ_BASECALLER_ALGO_VERSION
set(Baz_BASECALLER_ALGO_VERSION "5.0.0")

# Generate header with versions
configure_file(template/BazVersion.h.in ${CMAKE_CURRENT_BINARY_DIR}/generated/BazVersion.h)

add_library(libBazio
    STATIC
        BazCore.h
        BazEventData.cpp
        BazEventData.h
        BazReader.cpp
        BazReader.h
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
        RegionLabel.h
        RegionLabelType.h
        Sanity.h
        Simulation.cpp
        Simulation.h
        SimulationRNG.h
        SimulateWriteUtils.h
        SimulateWriteUtils.cpp
        SmartBazEnum.h
        SmartMemory.h
        SuperChunkMeta.h
        Timing.h
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
        libBazioEncoding
        libBazioFile
        libBazioWriting
        libPacbioCplusplusApi
        libPacbioSmrtdata
        libJsoncpp
        libHalf
)

add_subdirectory(bazviewer)
add_subdirectory(encoding)
add_subdirectory(file)
add_subdirectory(writing)
