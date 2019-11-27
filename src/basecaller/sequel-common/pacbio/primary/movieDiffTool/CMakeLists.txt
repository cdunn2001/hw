cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)

project(sequel_movie_diff)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")

  if(NOT TARGET sequel_movie_diff)
    add_executable(sequel_movie_diff
      main.cpp
    )

    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.."
      ${CMAKE_MODULE_PATH})

    include (pacbio-primary-common)
    include (pacbio-tbb)

    add_definitions(-DUSE_TBB)

    target_link_libraries(sequel_movie_diff
      libSequelCommon1
      libTbb
      )

  endif()

endif()

