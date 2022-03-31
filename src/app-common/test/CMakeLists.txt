include(pacbio-gtest)

add_library(libTestAppCommon OBJECT
    testBarrier.cpp
    testLocked.cpp
    testMemInfo.cpp
    testProgressMessage.cpp
)
    
target_link_libraries(libTestAppCommon
    PUBLIC
        lib-app-common
        libPacbioCplusplusApi
        libGtestWithMain
)

add_executable(testAppCommon
    testMain.cpp
    testMain.h
)

target_link_libraries(testAppCommon
    libTestAppCommon
)

add_test(testAppCommon testAppCommon --gtest_output=xml:testAppCommon.xml)
