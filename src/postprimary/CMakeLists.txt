cmake_minimum_required (VERSION 3.13)

include(pacbio-hdf5)
include(pacbio-jemalloc)
include(pacbio-zlib)
include(pacbio-boost)
include(pacbio-cplusplus-api)

# Main project paths

set(PPA_RootDir       ${CMAKE_CURRENT_LIST_DIR})
set(PPA_TestsDir      ${PPA_RootDir}/test)
set(PPA_ThirdPartyDir ${PPA_RootDir}/third-party)

###############################################################################

# Main library

add_library(libPpaBase STATIC
    adapterfinder/AdapterCorrector.cpp
    bam/BamCommentSideband.h
    bam/BamCommentSideband.cpp
    bam/BamProducer.h
    bam/BarcodeStrategy.h
    bam/ComplexResult.h
    bam/EventData.h
    bam/EventData.cpp
    bam/Platform.h
    bam/ResultPacket.h
    bam/ResultWriter.h
    bam/ResultWriter.cpp
    bam/RuntimeMetaData.h
    bam/SubreadContext.h
    bam/SubreadLabeler.h
    bam/SubreadLabeler.cpp
    bam/SubreadLabelerMetrics.h
    bam/SubreadSplitter.h
    bam/SubreadSplitter.cpp
    bam/ThreadSafeMap.h
    bam/Tools.h
    bam/Validation.h
    hqrf/BlockHQRegionFinder.cpp
    hqrf/ClassificationTree.cpp
    hqrf/HQRegionFinder.cpp
    hqrf/HQRegionFinderParams.cpp
    hqrf/SequelCrfParams.cpp
    hqrf/SpiderCrfParams.cpp
    hqrf/ZOffsetCrfParams.cpp
    hqrf/StaticHQRegionFinder.cpp
    insertfinder/InsertFinder.cpp
    stats/ProductivityMetrics.cpp
    stats/ZmwMetrics.cpp
    stats/ZmwStats.cpp
)



###############################################################################

# Build-time options

option(Baz2bam_Jemalloc "Builds baz2bam binary using jemalloc.  Defaults to on,\
                         but must be disabled if you wish to play with other allocators \
                         (e.g. using LD_PRELOAD to get tcmalloc or something)" ON)


if(CMAKE_COMPILER_IS_GNUCC)
  target_compile_options(libPpaBase PRIVATE "-fpermissive")
endif()
target_compile_definitions(libPpaBase
    PUBLIC
        -DSEQAN_NO_GLOBAL_EXCEPTION_HANDLER
)

set(DIAGNOSTICS True)
if ($ENV{NO_DIAGNOSTICS})
    set(DIAGNOSTICS False)
endif()

# Add adapter diagnostic verbosity
if (DIAGNOSTICS)
    target_compile_definitions(libPpaBase PUBLIC -DDIAGNOSTICS)
endif()

set(RTMETRICS_DIAGNOSTICS False)
if ($ENV{RTMETRICS_DIAGNOSTICS})
    set(RTMETRICS_DIAGNOSTICS True)
endif()

if (RTMETRICS_DIAGNOSTICS)
    target_compile_definitions(libPpaBase PUBLIC -DRTMETRICS_STSH5)
    target_compile_definitions(libSequelCommon1 PUBLIC -DRTMETRICS_STSH5)
endif()

# Include directories

target_include_directories(libPpaBase
    PUBLIC
        ${PPA_RootDir}/..
        ${PPA_ThirdPartyDir}
)

set(pbsparse_INCLUDE_DIRS ${PPA_ThirdPartyDir}/pbsparse/include)
include(pacbio-seqan)
include(pacbio-pbbam)
include(pacbio-spoa)

target_include_directories(libPpaBase SYSTEM
    PUBLIC
        ${pbsparse_INCLUDE_DIRS}
)

# Library dependencies

target_link_libraries(libPpaBase
    PUBLIC
        libHdf5Cpp
        libPacbioCplusplusApi
        libSequelCommon1
        libSeqan
        libPbbam
        libSpoa
)

# Additional third-party library dependencies

add_library(libPpaInterfaces INTERFACE)

target_include_directories(libPpaInterfaces
    INTERFACE
        ${pbbam_INCLUDE_DIRS}
)

target_link_libraries(libPpaInterfaces INTERFACE libPbbam)

###############################################################################

# Main applications

add_subdirectory(application)

###############################################################################

# Testing

enable_testing()
add_subdirectory(test)

###############################################################################

