include(pacbio-gtest)

add_library(libTestPaWs OBJECT
    testApi.cpp
    testPaWsConfig.cpp
    testPaWsProcess.cpp
    testWebService.cpp
    )
    
target_link_libraries(libTestPaWs
    PUBLIC
    lib-pa-ws
    libPacbioCplusplusApi
    libGtestWithMain
    )

add_executable(testPaWs
    testMain.cpp
    testMain.h
)

target_link_libraries(testPaWs
    libTestPaWs
    )

add_test(testPaWs testPaWs --gtest_output=xml:testPaWs.xml)
