#!/bin/bash

NUMA_NODE="${NUMA_NODE:-0}"
GPU_ID="${GPU_ID:-$(echo $NUMA_NODE)}"
OPTIONAL_DEBUGGER=${OPTIONAL_DEBUGGER:-}
# examples for running with optional debugging tools:
#  OPTIONAL_DEBUGGER="gdb --args" application/smrt-basecaller-launch.sh
#  OPTIONAL_DEBUGGER="valgrind" application/smrt-basecaller-launch.sh

dir=$(dirname $(realpath $0))

export CUDA_VISIBLE_DEVICES="${GPU_ID}"  
numactl --cpubind ${NUMA_NODE} --membind ${NUMA_NODE} ${OPTIONAL_DEBUGGER} ${dir}/smrt-basecaller "$@"
