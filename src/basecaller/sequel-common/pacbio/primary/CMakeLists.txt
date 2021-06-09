# This exports 3 basic targets
#  libSequelCommon - all library code and dependencies of that code
#  libSequelCommon1 - just library code, no dependencies

cmake_minimum_required (VERSION 3.13)

project (pacbio-sequel-common CXX)


option(BUILD_SEQUEL_COMMON_TESTS "If ON, the unit and functional tests are built")
option(SHM_REALTIME              "If ON, build the pipeline using PoolShmRealtime instead of PoolSCIF (needed for Purley Skylake with no MICS)")

# Allow use of std::min in Visual Studio.
add_definitions(-DNOMINMAX)

include(pacbio-cplusplus-api)
include(pacbio-hdf5)
include(pacbio-boost)
include(pacbio-jsoncpp)
include(pacbio-half)

# libSequelCommon1 does not include dependencies. Use libSequelCommon if you want the dependencies automatically linked
# in.

add_library(libSequelCommon1
    STATIC
        BasecallingMetrics.h
        BasecallingMetrics.cpp
        DetectionMode.h
        DetectionModeScalar.h
        DetectionModeScalar.cpp
        HDF5_type_traits.h
        HDF5_type_traits.cpp
        HDF5cpp.h
        HDFMultiArrayIO.h
        HDFMultiArrayIO.cpp
        HQRFMethod.h
        HQRFMethod.cpp
        HQRFParamsMCart.h
        HQRFParamsMCart.cpp
        HQRFParamsNCart.h
        HQRFParamsNCart.cpp
        PulseLookback.h
        SequelDefinitions.h
        SequelHDF5.h
        SequelHDF5.cpp
        TraceAnalysisMetrics.h
        Util.h
        ZmwReducedStatsFile.h
        ZmwReducedStatsFile.cpp
        ZmwStatsFile.h
        ZmwStatsFile.cpp
)

target_include_directories(libSequelCommon1
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/../..
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

target_link_libraries(libSequelCommon1
    PUBLIC
        libBazio
        libPacbioCplusplusApi
        libPacbioSmrtdata
        libHalf
        libHdf5Cpp
        libBoost
        libJsoncpp
)

if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
endif()

# The  libSequelCommon target includes ALL the full dependencies.
add_library(libSequelCommon
    STATIC
        ${CMAKE_CURRENT_BINARY_DIR}/null.cpp
)


message(STATUS "libSequelCommon boost dependencies: Boost_LIBRARIES= ${Boost_LIBRARIES}" )
target_link_libraries(libSequelCommon
    PUBLIC
        libSequelCommon1
    PRIVATE
        SegFault
)