# This exports 3 basic targets
#  libBazio - all library code and dependencies of that code
#  libSequelCommon - all library code and dependencies of that code
#  libSequelCommon1 - just library code, no dependencies

cmake_minimum_required (VERSION 2.8)
cmake_policy(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0022 OLD)

project (pacbio-sequel-common CXX)

option(BUILD_SEQUEL_COMMON_TESTS "If ON, the unit and functional tests are built")
option(SHM_REALTIME              "If ON, build the pipeline using PoolShmRealtime instead of PoolSCIF (needed for Purley Skylake with no MICS)")

# Allow use of std::min in Visual Studio.
add_definitions(-DNOMINMAX)


include(pacbio-hdf5)
include(pacbio-tbb)
include(pacbio-boost)
include(pacbio-jsoncpp)
#include(pacbio-smrtdata)

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

add_library(libBazio STATIC
    BazBuffer.h
    BazCore.h
    BazReader.cpp
    BazReader.h
    BazWriter.cpp
    BazWriter.h
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
    FileHeaderBuilder.cpp
    FileHeaderValidator.h
    MetricBlock.h
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
    SmartBazEnum.h
    SmartMemory.h
    SuperChunkMeta.h
    Timing.h
    ZmwSlice.h
    ZmwSliceHeader.h
)

include(pacbio-half)

target_include_directories(libBazio
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../..
        ${CMAKE_CURRENT_BINARY_DIR}/generated
#        ${half_INCLUDE_DIRS}
)

target_link_libraries(libBazio
    PUBLIC
        libPacbioCplusplusApi
        libPacbioSmrtdata
        libJsoncpp
        libHalf
)

# libSequelCommon1 does not include dependencies. Use libSequelCommon if you want the dependencies automatically linked
# in.

add_library( libSequelCommon1
  STATIC
#    Acquisition_Setup.h
#    Acquisition_Setup.cpp
#    AcquisitionProxy.h
#    AcquisitionProxy.cpp
#    AcquisitionRemoteControl.h
#    Alarm.h
#    Alarm.cpp
#    AnalogMode.h
#    BasecallerRemoteControl.h
#    BasecallingMetrics.h
#    BasecallingMetrics.cpp
#    BasecallingMetricsGlue.h
#    BazIOConfig.h
    ChipLayout.h
    ChipLayout.cpp
    ChipLayoutSequel.h
    ChipLayoutSequel.cpp
    ChipLayoutBenchy.cpp
    ChipLayoutBenchyDemo.cpp
    ChipLayoutRTO2.h
    ChipLayoutRTO2.cpp
    ChipLayoutRTO3.h
    ChipLayoutRTO3.cpp
    ChipLayoutSpider.h
    ChipLayoutSpider.cpp
    ChipLayoutSpider1.h
    ChipLayoutSpider1.cpp
    ChipLayoutSpider1p0NTO.h
    ChipLayoutSpider1p0NTO.cpp
#    ChunkFileSink.h
#    ChunkFileSource.h
#    Chunking.h
#    debug.cpp
#    debug.h
#    DetectionMode.h
#    DetectionModeScalar.h
#    DetectionModeScalar.cpp
#    Event.h
#    Event.cpp
    EventObject.h
    EventObject.cpp
    HDF5_type_traits.h
    HDF5_type_traits.cpp
    HDF5cpp.h
    HDFMultiArrayIO.h
    HDFMultiArrayIO.cpp
#    IntInterval.h
#    ipc_config.h
#    IPipelineHost.h
#    ITrancheNew.h
#    Kernel.h
#    Kernel.cpp
#    LaserPowerChange.h
#    LaserPowerChange.cpp
#    LiveViewFrame.h
#    ManuallyBufferedFile.h
#    ManuallyBufferedFile.cpp
#    MemoryBuffer.h
#    MemoryBuffer.cpp
#    PrimaryConfig.h
#    PulseLookback.h
    SequEl_4.0_RTO2.csv.gz.cpp
    SequEl_4.0_RTO3.csv.gz.cpp
#    SequelCalibrationFile.h
#    SequelCalibrationFile.cpp
#    SequelDefinitions.h
    SequelHDF5.h
    SequelHDF5.cpp
    SequelMovie.h
    SequelMovie.cpp
    SequelMovieEventsHDF5.h
    SequelMovieEventsHDF5.cpp
    SequelMovieFrame.h
    SequelMovieFrame.cpp
    SequelMovieFileHDF5.h
    SequelMovieFileHDF5.cpp
#    SequelMovieFileCRC.h
#    SequelMovieFileCRC.cpp
    SequelMovieFactory.h
    SequelMovieFactory.cpp
    SequelROI.h
    SequelROI.cpp
    SequelTraceFile.h
    SequelTraceFile.cpp
#    SmrtSensor.h
#    SmrtSensor.cpp
#    Sparse2DArray.h
#    Sparse2DArray.cpp
    Spider_1p0_NTO.h
    Spider_1p0_NTO.cpp
#    Statistic.h
#    TemporaryDirectory.h
#    TemporaryDirectory.cpp
#    Tile.h
#    TraceAnalysisMetrics.h
#    TraceFileSource.h
#    TrainedCartParams.h
#    TrainedCartParams.cpp
#    Tranche.h
#    Tranche.cpp
#    TrancheSizes.h
#    TrancheTitle.h
#    UnitCell.h
#    Util.h
#    WorkingFile.h
#    ZmwMetricsRecorder.h
#    ZmwMetricsRecorder.cpp
#    ZmwResultBuffer.h
#    ZmwReducedStatsFile.h
#    ZmwReducedStatsFile.cpp
#    ZmwStatsFile.h
#    ZmwStatsFile.cpp
#    ZmwStatsFileDefinition.h
#    ZmwStatsFileData.h
)


target_include_directories(libSequelCommon1
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../..
#        ${half_INCLUDE_DIRS}
)

target_include_directories(libSequelCommon1
    SYSTEM
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../../../../ThirdParty/halffloat/half-1.11.0/include/
)


target_compile_definitions(libSequelCommon1
    PUBLIC
        -DCMAKE_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\"
)

if(SHM_REALTIME)
    message(STATUS "Building SHM_REALTIME, for Purley Skylake and no MICS")
    target_compile_definitions(libSequelCommon1
            PUBLIC
            -DUSE_SHM_REALTIME=1
            )
endif(SHM_REALTIME)

# set( BOOST_ROOT ${PACBIO_THIRDPARTY} )
# set(Boost_USE_STATIC_LIBS ON)
include(pacbio-zlib)

target_link_libraries( libSequelCommon1
  PUBLIC
    libPacbioCplusplusApi
    libPacbioSmrtdata
    libHdf5_cpp
    ${Boost_LIBRARIES}  # boost::iostreams required for ChipLayout which uses gzip streams
    libJsoncpp
  INTERFACE
    libZlib
)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
endif()

# The  libSequelCommon target includes ALL the full dependencies.
add_library( libSequelCommon
    STATIC
        ${CMAKE_CURRENT_BINARY_DIR}/null.cpp
)


message(STATUS "libSequelCommon boost dependencies: Boost_LIBRARIES= ${Boost_LIBRARIES}" )
target_link_libraries( libSequelCommon
    PUBLIC
        libSequelCommon1
        libTbb
    PRIVATE
        SegFault
)

add_custom_target(rstdocs DEPENDS ZmwStats.rst)
add_custom_command(OUTPUT ZmwStats.rst
    COMMAND cpp -P ${CMAKE_CURRENT_LIST_DIR}/ZmwStats_template.rst | column --table --separator '|' --output-separator '|' > ZmwStats.rst
    DEPENDS
	${CMAKE_CURRENT_LIST_DIR}/ZmwStats_template.rst
	${CMAKE_CURRENT_LIST_DIR}/ZmwStatsFileDefinition.h
    COMMENT Making RST documentation for ZmwStats.h
)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_LIST_DIR} OR PA_build_test_sequel_common)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(bazviewer)
