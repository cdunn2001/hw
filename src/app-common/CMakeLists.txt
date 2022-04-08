
cmake_minimum_required (VERSION 3.8)
cmake_policy(VERSION 3.8)
cmake_policy(SET CMP0076 NEW)

project (app-common)

if(${CMAKE_CURRENT_LIST_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  enable_testing()

  include(../../setup-thirdparty.cmake)
  SetupProject("${CMAKE_CURRENT_LIST_DIR}/nexus-versions")
endif()


# The SOFTWARE_VERSION is used to create the RPM and installation directory, and it is reported
# by the application and logged.
set(SOFTWARE_VERSION "0.0.0")

add_subdirectory(test)

###########################################################
include(pacbio-versionheaderfile)
include(pacbio-cmake-config-vars)
include(pacbio-git-info)
include(pacbio-cplusplus-api)
###########################################################

generate_version_h(${SOFTWARE_VERSION})

add_library(lib-app-common
    STATIC
        Barrier.h
        Barrier.cpp
        HttpHelpers.h
        HttpHelpers.cpp
        Locked.h
        Locked.cpp
        MemInfo.h
        MemInfo.cpp
        ProgressMessage.h
        ProgressMessage.cpp
        ThreadController.h
)

target_link_libraries(lib-app-common
    PUBLIC
        libBoost
        libPacbioCplusplusApi
        curl
)

target_include_directories(lib-app-common
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/..
)
