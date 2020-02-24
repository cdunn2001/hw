set (VERSION bid13)

set (LOCAL_THIRD_PARTY_SCRIPTS off CACHE bool "Use local version of third party scripts" FORCE)
set (LOCAL_THIRD_PARTY_LOCATION ${CMAKE_CURRENT_LIST_DIR}/pa-third-party CACHE string "Location of a local  pa-third-party repository" FORCE)

function (SetupProject projName)

if (LOCAL_THIRD_PARTY_SCRIPTS)
    include(${LOCAL_THIRD_PARTY_LOCATION}/${projName}-setup.cmake)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)
    message("Setting module path to: " ${CMAKE_MODULE_PATH})
else()

    set(REPO_URL http://nexus.pacificbiosciences.com/repository/maven-thirdparty) # TODO: use https once it is available
    set(DOWNLOAD_TIMEOUT 60) # seconds
    set(LOCK_TIMEOUT 600) # seconds
    execute_process(COMMAND git rev-parse --show-toplevel OUTPUT_VARIABLE GIT_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(DEPCACHE ${GIT_ROOT}/depcache)

    set(ROOT ${DEPCACHE}/pa-versions/${VERSION}/${projName})
    FILE(MAKE_DIRECTORY ${DEPCACHE}/pa-versions/${VERSION})

    set(DEP_ARCHIVE_FILE pacbio-pa-third-party-${projName}-versions-${VERSION}-noarch.tar.gz)
    set(DEP_ARCHIVE ${DEPCACHE}/pa-versions/${DEP_ARCHIVE_FILE})
    set(DEP_LOCK_FILE ${DEPCACHE}/pa-versions.lock)
    FILE(LOCK ${DEP_LOCK_FILE} TIMEOUT ${LOCK_TIMEOUT})
    IF(NOT EXISTS "${ROOT}")
      set(URL ${REPO_URL}/pa-versions/pa-versions/${VERSION}/${DEP_ARCHIVE_FILE})
      message("Will download dependency pa-version to ${ROOT} from ${URL}")
      # Download the tarball
      FILE(REMOVE_RECURSE ${ROOT})
      FILE(REMOVE ${DEP_ARCHIVE} ${DEP_ARCHIVE}.md5)
      message("Downloading ${URL}.md5")
      FILE(DOWNLOAD 
        ${URL}.md5 ${DEP_ARCHIVE}.md5 
        INACTIVITY_TIMEOUT ${DOWNLOAD_TIMEOUT} 
        TIMEOUT ${DOWNLOAD_TIMEOUT} 
        STATUS DOWNLOAD_STATUS
      )
      FILE(STRINGS ${DEP_ARCHIVE}.md5 DEP_ARCHIVE_MD5 NO_HEX_CONVERSION)
      IF ("${DEP_ARCHIVE_MD5}" MATCHES "[a-f0-9]+")
        message("MD5 is ${DEP_ARCHIVE_MD5}")
      ELSE()
        message(FATAL_ERROR "Failed to download md5 from: ${URL}.md5; Status: ${DOWNLOAD_STATUS}; File is: ${DEP_ARCHIVE}.md5; MD5 is: ${DEP_ARCHIVE_MD5}")
      ENDIF()
      message("Downloading ${URL}")
      FILE(DOWNLOAD ${URL} ${DEP_ARCHIVE} 
        INACTIVITY_TIMEOUT ${DOWNLOAD_TIMEOUT} 
        TIMEOUT ${DOWNLOAD_TIMEOUT} 
        STATUS DOWNLOAD_STATUS 
        SHOW_PROGRESS 
        EXPECTED_MD5 ${DEP_ARCHIVE_MD5}
      )
      message("download archive status ${DOWNLOAD_STATUS}")

      execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${DEP_ARCHIVE}
        WORKING_DIRECTORY ${DEPCACHE}/pa-versions
      )
      FILE(RENAME ${DEPCACHE}/pa-versions/pacbio-pa-third-party-${projName}-versions-${VERSION}-noarch ${ROOT})
    ENDIF(NOT EXISTS "${ROOT}")
    FILE(LOCK "${DEP_LOCK_FILE}" RELEASE)
    IF(NOT EXISTS "${ROOT}")
      message(FATAL_ERROR "Expected dependency ${package} in ${ROOT}")
    ENDIF()

    include(${ROOT}/${projName}-setup.cmake)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} PARENT_SCOPE)
    message("Setting module path to: " ${CMAKE_MODULE_PATH})
endif()
endfunction(SetupProject)

# TODO this doesn't need to be a function...
SetupProject(mongo)
