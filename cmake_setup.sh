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


for d in \
   build/pa-ws/gcc/x86_64/Release \
   build/pa-cal/gcc/x86_64/Release
#   build/basecaller/gcc/x86_64/Release \
#   build/basecaller/gcc/x86_64/Debug \
#   build/basecaller/icpc/x86_64/Release \
#   build/basecaller/icpc/x86_64/Debug
do
    rm -rf $d 
    mkdir -p $d
    elems=(${d//\// })
    proj=${elems[1]}
    tool=${elems[2]}
    arch=${elems[3]}
    type=${elems[4]}
    echo ${elems[0]}  proj $proj tool $tool arch $arch type $type
    top=../../../../..
    tc=$top/TC-${tool}-cuda-${arch}.cmake

# Note: CMAKE_CUDA_HOST_COMPILER must be set here.  By the time we're in a cmake script (e.g. when parsing the toolchain file) it's already too late
#       and cuda will have at least partially latched on to whatever random host compiler it managed to find
    echo    cmake "-G${generator}" --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=${compilers[$tool]} -DCMAKE_BUILD_TYPE=$type -DCMAKE_TOOLCHAIN_FILE=$tc $top/src/$proj
    (cd $d; cmake "-G${generator}" --debug-trycompile -DCMAKE_CUDA_HOST_COMPILER=${compilers[$tool]} -DCMAKE_BUILD_TYPE=$type -DCMAKE_TOOLCHAIN_FILE=$tc $top/src/$proj)
done

