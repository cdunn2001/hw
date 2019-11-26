
option(LOCAL_THIRD_PARTY_SCRIPTS off)
set (LOCAL_THIRD_PARTY_LOCATION ${CMAKE_CURRENT_LIST_DIR}/pa-third-party CACHE string "Location of a local  pa-third-party repository")

if (LOCAL_THIRD_PARTY_SCRIPTS)
  include(${LOCAL_THIRD_PARTY_LOCATION}/mongo-setup.cmake)
else()
  message(FATAL_ERROR "nexus download not yet supported")
endif()
