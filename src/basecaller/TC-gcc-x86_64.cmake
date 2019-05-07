set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_SYSTEM_VERSION 1)

# Specify the cross compiler
set(CMAKE_C_COMPILER cc)
set(CMAKE_CXX_COMPILER c++)
# set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_CXX_FLAGS "-Wall -msse4.2 -Wno-missing-field-initializers -Wno-unused-local-typedefs -Wno-conversion-null -Wenum-compare -Wno-unknown-pragmas -DEIGEN_SIMD_SIZE=16 " CACHE STRING "")

set(CMAKE_CXX_FLAGS_DEBUG           "-O0 -g ${CMAKE_CXX_FLAGS_DEBUG_1}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -DNDEBUG "    CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O3 -g " CACHE STRING "")
