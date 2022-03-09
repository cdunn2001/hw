
cmake_minimum_required(VERSION 3.20)
project(MongoBasecaller
    VERSION 0.0
    LANGUAGES C CXX CUDA
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "70;80")

enable_testing()

include(../../setup-thirdparty.cmake)
SetupProject("${CMAKE_CURRENT_LIST_DIR}/nexus-versions")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# This is a bit of a hack.  Enabling cuda (above in the project command)
# will already cause these includes to be added to the compilation
# command.  PLUS nvcc automatically adds these paths when it forwards code
# to the host compiler, so it's already redundant.  However both those
# are done as regular includes, which causes a flood of warnings if strict
# warnings are enabled.  This adds it as a system path, which overries the
# non-system versions, though having the duplicate specifications may
# cause a warning of it's own (which can be individually suppressed)
include_directories(
    SYSTEM
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Find pacbio/common and primary/common cmake modules.
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_CURRENT_LIST_DIR}/"
)

# Used to define include directories for targets defined in subdirectories.
set(MONGO_PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})

include(pacbio-cplusplus-api)
include(pacbio-cmake-config-vars)

generate_config_header(mongo-config.h MongoConfig)

include(pacbio-hw-mongo)

add_subdirectory(sequel-common/pacbio/primary sequel-common)
add_subdirectory(../app-common app-common)
add_subdirectory(common)
add_subdirectory(applications)
add_subdirectory(appModules)
add_subdirectory(prototypes)
add_subdirectory(basecaller)
add_subdirectory(../bazio bazio)
add_subdirectory(dataTypes)
add_subdirectory(../postprimary postprimary)