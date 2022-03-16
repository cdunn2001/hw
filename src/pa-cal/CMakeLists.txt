cmake_minimum_required(VERSION 3.20)
cmake_policy(VERSION 3.8)
cmake_policy(SET CMP0076 NEW)

project (pa-cal)

enable_testing()

include(../../setup-thirdparty.cmake)
SetupProject("${CMAKE_CURRENT_LIST_DIR}/nexus-versions")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Used to define include directories for targets defined in subdirectories.
set(PRIMARY_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

include_directories(
    ${PRIMARY_PROJECT_DIR}/..
)

include(pacbio-eigen)
include(pacbio-cplusplus-api)
include(pacbio-cmake-config-vars)
include(pacbio-hw-mongo)

generate_config_header(kestrel-config.h PrimaryConfig)

add_subdirectory(../app-common app-common)


# The SOFTWARE_VERSION is used to create the RPM and installation directory, and it is reported
# by the application and logged.
set(SOFTWARE_VERSION "0.1.0")

# temporary hack, to work around some accidentally visible version strings in the hw-mongo
set(SOFTWARE_MAJOR_VERSION 0)
set(SOFTWARE_MINOR_VERSION 1)
set(SOFTWARE_REVISION_VERSION 0)
set(SOFTWARE_VERSION "${SOFTWARE_MAJOR_VERSION}.${SOFTWARE_MINOR_VERSION}.${SOFTWARE_REVISION_VERSION}")
add_definitions(
    -DSOFTWARE_MAJOR_VERSION=${SOFTWARE_MAJOR_VERSION}
    -DSOFTWARE_MINOR_VERSION=${SOFTWARE_MINOR_VERSION}
    -DSOFTWARE_REVISION_VERSION=${SOFTWARE_REVISION_VERSION}
)

add_subdirectory(test)

###########################################################
include(pacbio-versionheaderfile)
include(pacbio-cmake-config-vars)
include(pacbio-git-info)
###########################################################

generate_version_h(${SOFTWARE_VERSION})
message(STATUS "SOFTWARE_VERSION is ${SOFTWARE_VERSION}")
generate_config_header(PaCalHeader.h PaCalHeader)

add_library(lib-pa-cal
    STATIC
        ExitCodes.h
        FrameAnalyzer.h
        FrameAnalyzer.cpp
        SignalSimulator.h
        SignalSimulator.cpp
        PaCalConfig.h
        PaCalConfig.cpp
        PaCalConstants.h
        PaCalProcess.h
        PaCalProcess.cpp
)

target_link_libraries(lib-pa-cal
    PUBLIC
        libBoost
        libEigen
        libPacbioCplusplusApi
        libPacbioHwAcquisition
        GitVersionInfo
)

add_executable(pa-cal
    main.cpp
)

target_link_libraries(pa-cal
    PUBLIC
        lib-pa-cal
)


set(PREFIX /opt/pacbio/pa-cal-${SOFTWARE_VERSION})
set(CMAKE_INSTALL_PREFIX ${PREFIX})


###########################################################
# systemd and RPM
set(PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package)

# common RPM settings
set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_SUMMARY "PacBio Primary Analysis Calibration")
set(CPACK_PACKAGE_VERSION_MAJOR ${SOFTWARE_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${SOFTWARE_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${SOFTWARE_REVISION_VERSION})
set(CPACK_PACKAGE_VERSION       ${SOFTWARE_VERSION})
set(CPACK_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_SUMMARY})
set(CPACK_PACKAGE_CONTACT "support@pacbio.com")
set(CPACK_PACKAGE_VENDOR "Pacific Biosciences Ltd")
set(CPACK_PACKAGING_INSTALL_PREFIX ${PREFIX})
set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")
set(CPACK_RPM_SPEC_INSTALL_POST /bin/true)   # don't run 'strip' after installation
set(CPACK_RPM_COMPONENT_INSTALL "on")
set(CPACK_COMPONENTS_GROUPING "ONE_PER_GROUP")

set(SYSTEM_EXEC pa-cal)
set(SYSTEMD_ALIAS pacbio-pa-cal)

configure_file(${PACKAGE_DIR}/postInstall.sh.in            ${CMAKE_CURRENT_BINARY_DIR}/postInstall.sh           @ONLY)
configure_file(${PACKAGE_DIR}/preInstall.sh.in             ${CMAKE_CURRENT_BINARY_DIR}/preInstall.sh            @ONLY)
configure_file(${PACKAGE_DIR}/preUninstall.sh.in           ${CMAKE_CURRENT_BINARY_DIR}/preUninstall.sh          @ONLY)
configure_file(${PACKAGE_DIR}/pacbio-pa-cal.modulefile.in  ${CMAKE_CURRENT_BINARY_DIR}/pacbio-pa-cal.modulefile @ONLY)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/pa-cal                  DESTINATION bin )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pacbio-pa-cal.modulefile RENAME ${SOFTWARE_VERSION} DESTINATION etc/modulefiles/pacbio-pa-cal )


# cf http://schneide.wordpress.com/2013/02/11/build-a-rpm-package-using-cmake/
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/install)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/install)
else()
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_NAME_ROOT ${SYSTEMD_ALIAS})
set(CPACK_PACKAGE_VERSION_MAJOR ${SOFTWARE_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${SOFTWARE_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${SOFTWARE_REVISION_VERSION})
set(CPACK_PACKAGE_VERSION       ${SOFTWARE_VERSION})
set(CPACK_PACKAGE_NAME "${CPACK_PACKAGE_NAME_ROOT}-${CPACK_PACKAGE_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_SUMMARY})
set(CPACK_PACKAGE_CONTACT "support@pacbio.com")
set(CPACK_PACKAGE_VENDOR "Pacific Biosciences Ltd")
set(CPACK_PACKAGING_INSTALL_PREFIX ${PREFIX})
set(CPACK_RPM_PACKAGE_RELEASE       1)
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME_ROOT}-${CPACK_PACKAGE_VERSION}-${CPACK_RPM_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

set(CPACK_RPM_PACKAGE_AUTOREQPROV " no")
set(CPACK_RPM_PRE_INSTALL_SCRIPT_FILE   ${CMAKE_CURRENT_BINARY_DIR}/preInstall.sh)
set(CPACK_RPM_POST_INSTALL_SCRIPT_FILE  ${CMAKE_CURRENT_BINARY_DIR}/postInstall.sh)
set(CPACK_RPM_PRE_UNINSTALL_SCRIPT_FILE ${CMAKE_CURRENT_BINARY_DIR}/preUninstall.sh)
set(CPACK_RPM_SPEC_INSTALL_POST /bin/true)   # don't run 'strip' after installation

include(CPack)
