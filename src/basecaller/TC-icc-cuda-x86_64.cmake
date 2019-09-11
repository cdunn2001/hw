set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_SYSTEM_VERSION 1)

set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)

# * CUDA_API_PER_THREAD_DEFAULT_STREAM is the equivalent of `--default-stream per thread` for nvcc.  Necessary if any pure
#   C++ code is going to call any cuda runtime functions.
#
# * BOOST_LOG_BROKEN_CONSTANT_EXPRESSIONS triggers a workaround for certain versions of microsoft compilers that don't
#   properly handle constant expressions in nontype template parameters.  CUDA seems to have the same problem, so we 
#   manually invoke that to prevent issues.
set(CMAKE_CXX_FLAGS "-std=c++14 -Wcheck -w3 -wd304 -wd383 -wd424 -wd444 -wd981 -wd1418 -wd1572 -wd2282 -wd2960 -wd11074 -wd11076 -Wno-unknown-pragmas -xCORE-AVX512 -DPB_CORE_AVX512 -DEIGEN_SIMD_SIZE=64 -DCUDA_API_PER_THREAD_DEFAULT_STREAM -DBOOST_LOG_BROKEN_CONSTANT_EXPRESSIONS"
    CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG           "-O0 -g" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE         "-O3 -DNDEBUG "    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O3 -g " CACHE STRING "" FORCE)

# 177 Function declared but never referenced
# 2547 Header specified as both system and non-system include
# 1419 External declaration in primary source file
# 3346 Dynamic exception specifications are deprecated
# 82 Storage class is not first
# 869 Parameter was never referenced
set(CMAKE_CUDA_FLAGS                "-gencode arch=compute_70,code=sm_70 --expt-relaxed-constexpr --default-stream per-thread --compiler-options=\"${CMAKE_CXX_FLAGS}\" --compiler-options=\"-wd2547 -wd177 -wd1419 -wd3346 -wd82 -wd869\"" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELEASE        "-O3 -DNDEBUG -lineinfo" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_DEBUG          "-O0 -g -G" CACHE STRING "" FORCE)
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g -lineinfo" CACHE STRING "" FORCE)

set(CMAKE_LINKER_FLAGS_INIT         "WUB WUB")
