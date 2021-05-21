#!/bin/bash -e

distclean=${distclean:-0}  # set to 1 if you want to obliterate all previous build artifacts
generator=${generator:-Unix Makefiles}

if [[ $distclean != 0 ]]
then
    echo Removing depcache
    rm -rf ../../depcache
fi


rm -rf build/x86_64/Debug
rm -rf build/x86_64/Release
rm -rf build/x86_64_gcc/Debug
rm -rf build/x86_64_gcc/Release

mkdir -p build/x86_64/Debug
mkdir -p build/x86_64/Release
mkdir -p build/x86_64_gcc/Debug
mkdir -p build/x86_64_gcc/Release

(cd build/x86_64/Debug;       LDFLAGS="-Wl,--as-needed -static-libgcc -static-libstdc++" cmake "-G${generator}" --debug-trycompile -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_TOOLCHAIN_FILE=../../../../../TC-icc-x86_64.cmake ../../..)
(cd build/x86_64/Release;     LDFLAGS="-Wl,--as-needed -static-libgcc -static-libstdc++" cmake "-G${generator}" --debug-trycompile -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../../TC-icc-x86_64.cmake ../../..)
(cd build/x86_64_gcc/Debug;   LDFLAGS="-Wl,--as-needed -static-libgcc -static-libstdc++" cmake "-G${generator}" --debug-trycompile -DCMAKE_BUILD_TYPE=Debug   -DCMAKE_TOOLCHAIN_FILE=../../../../../TC-gcc-x86_64.cmake ../../..)
(cd build/x86_64_gcc/Release; LDFLAGS="-Wl,--as-needed -static-libgcc -static-libstdc++" cmake "-G${generator}" --debug-trycompile -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=../../../../../TC-gcc-x86_64.cmake ../../..)
