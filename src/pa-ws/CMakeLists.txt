cmake_minimum_required (VERSION 3.13)
cmake_policy(VERSION 3.8)
cmake_policy(SET CMP0076 NEW)

project (pa-ws
    VERSION 0.0
    LANGUAGES C CXX
)

enable_testing()

list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/nexus-versions"
)

include(../../setup-thirdparty.cmake)

# TODO: Use C++17 when we have a newer version of GCC available.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Used to define include directories for targets defined in subdirectories.
set(PRIMARY_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

include_directories(
    ${PRIMARY_PROJECT_DIR}/..
)

include(pacbio-cplusplus-api)
include(pacbio-cmake-config-vars)

generate_config_header(kestrel-config.h PrimaryConfig)

add_subdirectory(../common common)
add_subdirectory(../../doc doc)


# The SOFTWARE_VERSION is used to create the RPM and installation directory, and it is reported
# by the application and logged.
set(SOFTWARE_VERSION "0.0.0")
set(SYSTEMD_COMMON_JSON "/etc/pacbio/app-common.json")

add_subdirectory(dashboard)
add_subdirectory(test)

###########################################################
include(pacbio-versionheaderfile)
include(pacbio-cmake-config-vars)
include(pacbio-git-info)
###########################################################

generate_version_h(${SOFTWARE_VERSION})
message(STATUS "SOFTWARE_VERSION is ${SOFTWARE_VERSION}")
generate_config_header(PaWsHeader.h PaWsHeader)
create_git_rev_header()

# # exported header-only library. To be used as an exported artifact.
# add_library(lib-wx-api INTERFACE)
# target_include_directories(lib-wx-api
#     INTERFACE
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/WxIpcApi/>
# )

# target_sources(lib-wx-api
#     INTERFACE
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/WxIpcApi/WxIpcApi.h>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/WxIpcApi/WxRestApi.h>
#         $<INSTALL_INTERFACE:WxIpcApi/install/WxIpcApi.h>
#         $<INSTALL_INTERFACE:WxIpcApi/install/WxRestApi.h>
# )

add_library(lib-pa-ws
    STATIC
        ExitCodes.h
        mockup.cpp
        PaWsConfig.h
        PaWsConfig.cpp
        PaWsConstants.h
        PaWsProcess.h
        PaWsProcess.cpp
        WebService.h
        WebService.cpp

        # this file doesn't exist but is necessary to put here to force refreshing the git-rev.h file
        git-rev-pseudo-header.h
)

target_link_libraries(lib-pa-ws
    PUBLIC
      libBoost
      libPacbioCplusplusApi
      libPaWsDoc
      libPaWsDashboard
      curl
)

add_executable(pa-ws
    main.cpp
)

target_link_libraries(pa-ws
  PUBLIC
    lib-pa-ws
)


set(PREFIX /opt/pacbio/pa-ws-${SOFTWARE_VERSION})
set(CMAKE_INSTALL_PREFIX ${PREFIX})


###########################################################
# systemd and RPM
set(SYSTEMD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/systemd)
set(PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package)

# common RPM settings
set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_SUMMARY "PacBio Primary Analysis Webservice")
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

set(SYSTEM_EXEC pa-ws)
set(SYSTEMD_ALIAS pacbio-pa-ws)
set(SYSTEMD_SERVICE ${SYSTEMD_ALIAS}-${SOFTWARE_VERSION}.service)
# set(SYSTEMD_DEPENDENCIES "wolverine.service")
set(SYSTEMD_CONF_PATH /etc/pacbio/${SYSTEMD_ALIAS}.conf)

configure_file(${SYSTEMD_DIR}/pacbio-pa-X.service.in      ${CMAKE_CURRENT_BINARY_DIR}/${SYSTEMD_SERVICE}     @ONLY)
configure_file(${SYSTEMD_DIR}/pacbio-pa-X.conf.in         ${CMAKE_CURRENT_BINARY_DIR}/${SYSTEMD_ALIAS}.conf  @ONLY)
configure_file(${PACKAGE_DIR}/postInstall.sh.in           ${CMAKE_CURRENT_BINARY_DIR}/postInstall.sh         @ONLY)
configure_file(${PACKAGE_DIR}/preInstall.sh.in            ${CMAKE_CURRENT_BINARY_DIR}/preInstall.sh          @ONLY)
configure_file(${PACKAGE_DIR}/preUninstall.sh.in          ${CMAKE_CURRENT_BINARY_DIR}/preUninstall.sh        @ONLY)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/pa-ws                  DESTINATION bin )
install(PROGRAMS ${SYSTEMD_DIR}/precheck-pa-ws.sh                   DESTINATION bin )
install(FILES    ${CMAKE_CURRENT_BINARY_DIR}/${SYSTEMD_SERVICE}     DESTINATION systemd )
install(FILES    ${CMAKE_CURRENT_BINARY_DIR}/${SYSTEMD_ALIAS}.conf  DESTINATION systemd )

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
