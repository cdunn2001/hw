
cmake_minimum_required(VERSION 3.20)
include(pacbio-git-info)

set(SOFTWARE_MAJOR_VERSION 0)
set(SOFTWARE_MINOR_VERSION 0)
set(SOFTWARE_REVISION_VERSION 0)
set(SOFTWARE_VERSION "${SOFTWARE_MAJOR_VERSION}.${SOFTWARE_MINOR_VERSION}.${SOFTWARE_REVISION_VERSION}")


add_executable(smrt-basecaller
        SmrtBasecaller.cpp
)

configure_file(numa_launch.sh numa_launch.sh COPYONLY)
configure_file(4x_launch.sh 4x_launch.sh COPYONLY)

target_compile_definitions(smrt-basecaller
PUBLIC
        -DSOFTWARE_MAJOR_VERSION=${SOFTWARE_MAJOR_VERSION}
        -DSOFTWARE_MINOR_VERSION=${SOFTWARE_MINOR_VERSION}
        -DSOFTWARE_REVISION_VERSION=${SOFTWARE_REVISION_VERSION}
        -DSTRINGIFIED_SOFTWARE_VERSION="${SOFTWARE_VERSION}"
)

target_link_libraries(smrt-basecaller
    PRIVATE
        libAppModules
        libPacbioHwAcquisition
        GitVersionInfo
)
add_subdirectory(test)

###########################################################
# systemd and RPM
set(PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/package)
set(PREFIX /opt/pacbio/pa-smrt-basecaller-${SOFTWARE_VERSION})
set(CMAKE_INSTALL_PREFIX ${PREFIX})

# common RPM settings
set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_SUMMARY "PacBio Primary Analysis Basecaller")
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

set(SYSTEMD_COMMON_JSON /etc/pacbio/kestrel-common.json)  # TODO remove Kestrel name from this file.
set(SYSTEMD_ALIAS pacbio-pa-smrt-basecaller)

configure_file(${PACKAGE_DIR}/postInstall.sh.in              ${CMAKE_CURRENT_BINARY_DIR}/postInstall.sh         @ONLY)
configure_file(${PACKAGE_DIR}/preInstall.sh.in               ${CMAKE_CURRENT_BINARY_DIR}/preInstall.sh          @ONLY)
configure_file(${PACKAGE_DIR}/preUninstall.sh.in             ${CMAKE_CURRENT_BINARY_DIR}/preUninstall.sh        @ONLY)
configure_file(${PACKAGE_DIR}/pacbio-pa-smrt-basecaller.modulefile.in  ${CMAKE_CURRENT_BINARY_DIR}/pacbio-pa-smrt-basecaller.modulefile  @ONLY)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/smrt-basecaller          DESTINATION bin )
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/pacbio-pa-smrt-basecaller.modulefile RENAME ${SOFTWARE_VERSION} DESTINATION etc/modulefiles/pacbio-pa-smrt-basecaller )

#    set_target_properties(${UNBUNDLED_EXE} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

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
