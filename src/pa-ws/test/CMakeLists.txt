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

add_test(
    NAME testPaWs 
    COMMAND testPaWs --gtest_output=xml:testPaWs.xml)

add_test(
    NAME testPythonUnits
    COMMAND e2e_ve/bin/python3 -m unittest -v ${CMAKE_CURRENT_LIST_DIR}/kpa-ws-test.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

add_test(
    NAME testPaWsWithPythonBist
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/kpa-ws-test.py --bist
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)
