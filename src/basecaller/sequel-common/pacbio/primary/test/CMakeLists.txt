cmake_minimum_required(VERSION 3.2)
cmake_policy(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)

project(SequelCommonTest)


set(exec     testSequelCommon)
set(slowTest testSequelCommon_FunctionalTest)
set(EXE3     testSequelAcquisitionPipelineHost)

set(CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../common/pacbio-cplusplus-api"
    "${CMAKE_CURRENT_LIST_DIR}/../../../../../common/pacbio-testing-api"
    ${CMAKE_MODULE_PATH}
)
include(primary-config)
create_config_header(pacbio-primary-test-config.h CommonTest)
include(pacbio-cplusplus-gtest)
include(pacbio-primary-common)
include_directories("..")


set(SOURCE
    testAcquisition_Setup.cpp
    testAcquisitionProxy.cpp
    testAcquisitionRemoteControl.cpp
    testAlarm.cpp
    testAnalogMode.cpp
    testChipLayout.cpp
    testChunkFileSource.cpp
    testCRCFile.cpp
    testEvent.cpp
    testEventObject.cpp
#    ../FromAcquisition.cpp
#    testFromAcquisition.cpp
    TestHDFMultiArrayIO.cpp
    testIntInterval.cpp
    testKernel.cpp
    testLaserPowerChange.cpp
    testManuallyBufferedFile.cpp
    testMemoryBuffer.cpp
    testMetrics.cpp
#    testPipelineHost.cpp    TODO FIXME does not build
    testPrimaryConfig.cpp
    testSequelCalibrationFile.cpp
    testSequelMovie.cpp
    testSequelMovieFactory.cpp
    testSequelMovieFileHDF5.cpp
    testSequelROI.cpp
    testSequelTraceFileHDF5.cpp
    testSmrtSensor.cpp
    testSparse2DArray.cpp
    testTile.cpp
    testTraceFilePath.cpp
    testTranche.cpp
    testTrancheTitle.cpp
    testZmwReducedStatsFile.cpp
    testZmwStatsFile.cpp

    # BAZ-related test code
    testCodec.cpp
    testEmptyChunk.cpp
    testEndToEndBaz.cpp
    testEndToEndInternal.cpp
    testEndToEndNoBases.cpp
    testEndToEndNoMetrics.cpp
    testEndToEndRandMetrics.cpp
    testFastaParser.cpp
    testFileHeader.cpp
    testFileHeaderBuilder.cpp
    testFileHeaderInternal.cpp
    testLookupTable.cpp
    testMetricBlock.cpp
    testProductionDoubleZmw.cpp
    testProductionNoContent.cpp
    testProductionOnlyPulses.cpp
    testProductionWithPulse.cpp
    testReadBuffer.cpp
)

if(NOT TARGET ${exec})
    message(STATUS "Configuring ${exec}")

    add_library(libSequelCommonTest
        ${SOURCE}
    )

    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
	    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
    endif()

    add_executable(${exec}
        ${CMAKE_CURRENT_BINARY_DIR}/null.cpp
    )

    target_link_libraries( libSequelCommonTest
        libSequelCommon
        libBazio
    )

    target_link_libraries(${exec}
        -Wl,--whole-archive
        libSequelCommonTest
        -Wl,--no-whole-archive
        gtest_main
    )

    add_custom_command(TARGET ${exec}
        POST_BUILD
        COMMAND patchelf --set-rpath /opt/intel/tbb/lib/intel64/gcc4.4:/opt/micron/lib:/lib64:\\$$ORIGIN ${exec}
    )

    add_test(${exec}  ${exec} --gtest_output=xml:${exec}.xml)

endif()

if(NOT TARGET ${slowTest})
    message(STATUS "Configuring ${slowTest}")

    add_executable(${slowTest}
        ${SOURCE}
        testSequelTraceFileHDF5_FunctionalTest.cpp
    )

    target_link_libraries(${slowTest}
        libSequelCommonTest
        gtest_main
    )

    add_test(${slowTest}  ${slowTest} --gtest_output=xml:${slowTest}.xml)
endif()



if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "k1om")
    add_custom_target(runmicTestSequelCommon
        DEPENDS ${exec}

        COMMAND scp -i ${PBI_ID_RSA} ${CMAKE_CURRENT_SOURCE_DIR}/test_64[512]x2k[512].int16 pbi@mic0:
        COMMAND scp -i ${PBI_ID_RSA} ${CMAKE_CURRENT_SOURCE_DIR}/UnitTest.h5 pbi@mic0:
        COMMAND scp -i ${PBI_ID_RSA} ${exec} pbi@mic0:
        COMMAND ssh -i ${PBI_ID_RSA} pbi@mic0 chmod u+w test_64[512]x2k[512].int16 UnitTest.h5 ${exec}
        COMMAND ssh -i ${PBI_ID_RSA} pbi@mic0 LD_LIBRARY_PATH=/home/pbi /home/pbi/${exec}
     )
endif()
