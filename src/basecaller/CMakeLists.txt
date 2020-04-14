
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

# need to load mongo-tbb first. pacbio-primary-common
# is going to load primary-tbb, which currently has a
# slight problem where it tries to make us use
# /opt/intel/tbb, but we still actually take headers from
# the actual compiler installation, which is a different
# version of tbb.  Using this cmake file first will make
# primary-tbb do essentially nothing, and we just use both
# headers and libraries from the compiler installation.
# once the build system gets fixed mongo-tbb.cmake can be
# deleted
include(mongo-tbb)

include(pacbio-cplusplus-api)
include(mongo-config)

create_config_header(mongo-config.h MongoConfig)

if (true)
    list(APPEND CMAKE_MODULE_PATH
            /home/UNIXHOME/mlakata/git/hw-mongo/bamboo/install/gcc_x86_64/lib/CMakeTargets/
            )
    include(pacbio-hdf5)
    include(MongoDatasource)
    set_target_properties(HwMongo::libMongoDatasource PROPERTIES IMPORTED_GLOBAL TRUE)
    add_library(libMongoDatasource ALIAS HwMongo::libMongoDatasource)

    #    include(SequelCommon)
    add_subdirectory(sequel-common/pacbio/primary sequel-common)
else()
    add_subdirectory(sequel-common/pacbio/primary sequel-common)
endif()

add_subdirectory(common)
add_subdirectory(applications)
add_subdirectory(basecaller)
add_subdirectory(dataTypes)
add_subdirectory(prototypes)
