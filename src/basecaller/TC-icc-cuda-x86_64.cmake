set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)

# CUDA_API_PER_THREAD_DEFAULT_STREAM is the equivalent of `--default-stream per thread` for nvcc.  Necessary if any pure
# C++ code is going to call any cuda runtime functions.
set(CMAKE_CXX_FLAGS "-std=c++14 -Wcheck -w3 -wd304,383,424,444,981,1418,1572,2960,11074,11076 -Wno-unknown-pragmas -xCORE-AVX512 -DPB_CORE_AVX512 -DEIGEN_SIMD_SIZE=64 -DCUDA_API_PER_THREAD_DEFAULT_STREAM"
    CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG           "-O0 -g" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -DNDEBUG "    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O3 -g " CACHE STRING "" FORCE)

set(CMAKE_CUDA_FLAGS                "-gencode arch=compute_70,code=sm_70 --expt-relaxed-constexpr --default-stream per-thread" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG          "-O0 -g -G" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -lineinfo" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE        "-O3 -DNDEBUG -lineinfo" CACHE STRING "" FORCE)
