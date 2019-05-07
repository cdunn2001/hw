#!/bin/bash -e

distclean=${distclean:-0}  # set to 1 if you want to obliterate all previous build artifacts

if [[ $distclean != 0 ]]
then
    echo Removing depcache
    rm -rf ../../../depcache
fi


rm -rf build/x86_64/Debug_gcc
rm -rf build/x86_64/Release_gcc

mkdir -p build/x86_64/Debug_gcc
mkdir -p build/x86_64/Release_gcc

#(cd build/arm_64/Debug;       cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/aarch64-linux-gnu-g++-7 -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_TOOLCHAIN_FILE=../../../common/TC-gcc-cuda-arm.cmake ../../..)
#(cd build/arm_64/Release;     cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/aarch64-linux-gnu-g++-7 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../common/TC-gcc-cuda-arm.cmake ../../..)
(cd build/x86_64/Debug_gcc;       cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/gcc -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_TOOLCHAIN_FILE=../../../common/TC-gcc-cuda-x86_64.cmake ../../..)
(cd build/x86_64/Release_gcc;     cmake --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/devtoolset-6/root/usr/bin/gcc -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../common/TC-gcc-cuda-x86_64.cmake ../../..)

