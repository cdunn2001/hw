#!/bin/bash -e

distclean=${distclean:-0}  # set to 1 if you want to obliterate all previous build artifacts

if [[ $distclean != 0 ]]
then
    echo Removing depcache
    rm -rf ../depcache
fi


rm -rf build/x86_64/Debug_gcc
rm -rf build/x86_64/Release_gcc
rm -rf build/x86_64/Debug
rm -rf build/x86_64/Release

mkdir -p build/x86_64/Debug_gcc
mkdir -p build/x86_64/Release_gcc
mkdir -p build/x86_64/Debug
mkdir -p build/x86_64/Release

# Note: CMAKE_CUDA_HOST_COMPILER must be set here.  By the time we're in a cmake script (e.g. when parsing the toolchain file) it's already too late
#       and cuda will have at least partially latched on to whatever random host compiler it managed to find
(cd build/x86_64/Debug_gcc;       cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=gcc -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_TOOLCHAIN_FILE=TC-gcc-cuda-x86_64.cmake ../../..)
(cd build/x86_64/Release_gcc;     cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=TC-gcc-cuda-x86_64.cmake ../../..)
(cd build/x86_64/Debug;           cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=icpc -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_TOOLCHAIN_FILE=TC-icc-cuda-x86_64.cmake ../../..)
(cd build/x86_64/Release;         cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=icpc -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=TC-icc-cuda-x86_64.cmake ../../..)

