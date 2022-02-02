distclean=${distclean:-0}  # set to 1 if you want to obliterate all previous build artifacts
generator=${generator:-Ninja}
declare -A compilers
compilers["gcc"]=gcc
compilers["icc"]=icpc

if [[ $distclean != 0 ]]
then
    echo Removing depcache
    rm -rf depcache
fi


# the build directories must be 5 elements deep, with the convention:
#
#  build/$app/$toolkit/$arch/$type
#

for d in \
   build/pa-ws/gcc/x86_64/Release \
   build/pa-cal/gcc/x86_64/Release \
   build/common/gcc/x86_64/Release \
   build/basecaller/gcc/x86_64/Release \
   build/basecaller/gcc/x86_64/RelWithDebInfo \
   build/basecaller/gcc/x86_64/Debug
do
    rm -rf $d 
    mkdir -p $d
    elems=(${d//\// })  # split the path into the constituent elements
    proj=${elems[1]}
    tool=${elems[2]}
    arch=${elems[3]}
    type=${elems[4]}
    echo ${elems[0]} proj: $proj tool: $tool arch: $arch type: $type
    top=../../../../..
    tc=$top/TC-${tool}-cuda-${arch}.cmake

    pushd $d > /dev/null

    fulldir=$(pwd)

    # This generated script will blow away all files in the build directory (except itself) before running cmake. Unless the command line
    # arguments change, or there is severe disk corruption causing permissions problems, this local script should
    # be sufficient to start fresh with cmake.  Note that the `rm` command specifically does not use the -f force option,
    # since all files should be R/W.

    # Note: the CMAKE_CUDA_HOST_COMPILER variable must be set here on the command line for our particular version of cmake (3.13).
    #       It is not allowed in a toolchain file, *.cmake file or CMakeLists.txt file.
    #       (An upgrade to cmake could allow the -DCMAKE_CUDA_HOST_COMPILER argument be moved into a tool chain file.)

    cat <<HERE > cmake_setup.sh
#! /bin/bash
shopt -s extglob # allow the !(filename) glob syntax
cd ${fulldir}
files=\$(shopt -s nullglob; echo !(cmake_setup.sh|build.ninja))
if [[ \$files != "" ]]
then
    echo -n "Directory contains a build already, are you sure you want to delete all existing build files in $fulldir? (y/n)"
    read confirmation
    if [[ \$confirmation != "y" && \$confirmation != "yes" ]]
    then
        echo "not continuing"
        exit 1
    fi
    cd ${fulldir} && rm -rv !("cmake_setup.sh")
fi
cmake "-G${generator}" --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=${compilers[$tool]} -DCMAKE_BUILD_TYPE=$type -DCMAKE_TOOLCHAIN_FILE=$tc $top/src/$proj
HERE
    chmod +x cmake_setup.sh

    if [[ ${generator} == "Ninja" ]] ; then
# This original ninja file is a bootstrap to create the real ninja file. Once ninja runs once, it
# will delete this file.
      cat <<NINJA > build.ninja
rule run_cmake_setup
  command = ./cmake_setup.sh > \$out
build build.ninja : run_cmake_setup
NINJA
    fi

    popd > /dev/null
done
