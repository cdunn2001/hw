cmake_minimum_required(VERSION 2.8)

# this variable will be set to the full path where this file (primary-config.cmake) exists.
set(MongoBaseDir ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "Path to top of Mongo tree")
get_filename_component(workspaceDir ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)

include(pacbio-versionheaderfile)

# The function create_config_header() will create a "Config.h" file in the binary output
# folder and set the include_directories() to be able to find it. The Config.h is derived
# from the Config.h.in file and contains many useful CMAKE related paths and settings, in case
# your code needs to reflect back in to the source tree to grab test vector files, for example.

function(create_config_header)
  if (${ARGC} GREATER 1)
      list(GET ARGN 1 configNamespace)
  else()
      set(configNamespace Configuration)
  endif()
  if (${ARGC} GREATER 0)
     list(GET ARGN 0 header_file_name)
  else()
     set(header_file_name Config.h)
  endif()
  get_git_full_commit_info(GIT_LONG_COMMIT_HASH GIT_COMMIT_DATE)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/config)
  message(STATUS "Generating ${CMAKE_CURRENT_BINARY_DIR}/config/${header_file_name} with MongoBaseDir = ${MongoBaseDir} and workspaceDir = ${workspaceDir}")
  configure_file("${MongoBaseDir}/Config.h.in" ${CMAKE_CURRENT_BINARY_DIR}/config/${header_file_name})
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/config)
endfunction(create_config_header)

