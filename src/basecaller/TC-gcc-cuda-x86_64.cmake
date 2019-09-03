set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_C_COMPILER cc)
set(CMAKE_CXX_COMPILER c++)

# CUDA_API_PER_THREAD_DEFAULT_STREAM is the equivalent of `--default-stream per thread` for nvcc.  Necessary if any pure
# C++ code is going to call any cuda runtime functions.
set(CMAKE_CXX_FLAGS "-Wall -msse4.2 -Wno-missing-field-initializers -Wno-unused-local-typedefs -Wno-conversion-null -Wenum-compare -Wno-unknown-pragmas -DEIGEN_SIMD_SIZE=16 -DCUDA_API_PER_THREAD_DEFAULT_STREAM" CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG           "-O0 -g ${CMAKE_CXX_FLAGS_DEBUG_1}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -DNDEBUG "    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O3 -g " CACHE STRING "" FORCE)

set(CMAKE_CUDA_FLAGS                "-gencode arch=compute_70,code=sm_70 --expt-relaxed-constexpr --default-stream per-thread --compiler-options=\"${CMAKE_CXX_FLAGS}\"" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE        "-O3 -DNDEBUG -lineinfo" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG          "-O0 -g -G" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -lineinfo" CACHE STRING "" FORCE)
