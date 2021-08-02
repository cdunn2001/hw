include(pacbio-zlib)

include(pacbio-cmake-config-vars)
generate_config_header(test_data_config.h PaPpaTestConfig)

include(pacbio-gtest)

set(exec testSequelPpa)

set(SOURCE
    ReadSimulator.cpp
    testBamCommentSideband.cpp
    testReadType.cpp
    testZmwMetrics.cpp
    testStatsH5.cpp
    testHist.cpp
    testHQRegionFinderParams.cpp
    testInsertFinder.cpp
    testCrfHQRegionFinder.cpp
    testHQRegionFinder.cpp
    testMetadataParser.cpp
    testAdapter.cpp
    testControl.cpp
    testBamPAVerify.cpp
    testResultWriter.cpp
    testConfigMetadata.cpp
)

if(NOT TARGET ${exec})
    message(STATUS "Configuring ${exec}")

    add_library(libtestSequelPpa
        ${SOURCE}
    )

    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
    endif()

    add_executable(${exec}
        ${CMAKE_CURRENT_BINARY_DIR}/null.cpp
    )

    target_link_libraries(libtestSequelPpa
        libPpaBase
        libPpaInterfaces
        libSequelCommon1
        libGtest
    )

    target_link_libraries(${exec}
        -Wl,--whole-archive
        libtestSequelPpa
        -Wl,--no-whole-archive
        libGtestWithMain
    )

    add_test(${exec}  ${exec} --gtest_output=xml:${exec}.xml)

endif()
