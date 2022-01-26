#!/usr/bin/env bash

set -e

if [ ${VERBOSE+x} ]; then
    set -x
fi

usage()
{
    cat << EOF
    Usage: $0 [options] <COMPONENT>

    Components are:

    basecaller
    webservices
    calibration

    OPTIONS:
        -h  Show this message
        -b  Build only do not run test
        -t  Build type (default 'Release')
EOF
}

BUILD_ONLY=0
BUILD_TYPE=Release

while getopts ":bt:h?" OPTION
do
        case $OPTION in
                b)
                  BUILD_ONLY=1
                  ;;
		t)
                  BUILD_TYPE=$OPTARG
                  ;;
                h)
                  usage
                  exit
                  ;;
                ?)
                  usage
                  exit
                  ;;
        esac
done
shift $((OPTIND-1))

component=$1

kestrelRoot=$(dirname $(realpath $0 ))
. ${kestrelRoot}/module_setup.sh
module list

makeTarget()
{
    build_dir=$1/${BUILD_TYPE}
    shift
    targets=$@

    # Setup project
    ./cmake_setup.sh
    pushd ${build_dir}
    ./cmake_setup.sh

    cmake --build ./ -- $targets
    exitStatus=$?
    if [[ ${exitStatus} -ne 0 ]]; then
        exit $exitStatus
    fi
}

case ${component} in
basecaller)

    makeTarget build/basecaller/gcc/x86_64

    if [[ ${BUILD_ONLY} -eq 0 ]]; then
        ctest -VV --no-compress-output -T Test
    fi
    ;;
webservices)

    makeTarget build/pa-ws/gcc/x86_64

    if [[ ${BUILD_ONLY} -eq 0 ]]; then
        ctest -VV --no-compress-output -T Test
    fi
    ;;
calibration)

    makeTarget build/pa-cal/gcc/x86_64

    if [[ ${BUILD_ONLY} -eq 0 ]]; then
        ctest -VV --no-compress-output -T Test
    fi
    ;;
esac
