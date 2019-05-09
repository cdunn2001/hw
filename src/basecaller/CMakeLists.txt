
cmake_minimum_required(VERSION 3.13)
project(MongoBasecallerDemo
    VERSION 0.0
    LANGUAGES CXX CUDA
)

# TODO: Use C++17 when we have a newer version of GCC available.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Note: C++14 is the latest allowed by CUDA.  Even if we bump
# the standard for cpp files, we cannot here
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Find pacbio/common and primary/common cmake modules.
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/../common/pacbio-cplusplus-api"
    "${CMAKE_CURRENT_LIST_DIR}/../Sequel/common/"
    # "${CMAKE_CURRENT_LIST_DIR}/../../common/cmake-modules"
    # "${CMAKE_CURRENT_LIST_DIR}/../common/"
)

# Used to define include directories for targets defined in subdirectories.
set(MONGO_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

include(pacbio-cplusplus-api)
include(pacbio-primary-common)

add_subdirectory(common)
add_subdirectory(applications)
add_subdirectory(basecaller)
add_subdirectory(dataTypes)
add_subdirectory(prototypes)
