#!/bin/bash

NUMA_NODE="${NUMA_NODE:-0}"
GPU_ID="${GPU_ID:-$(echo $NUMA_NODE)}"

dir=$(dirname $0)

CUDA_VISIBLE_DEVICES="${GPU_ID}"  numactl --cpubind ${NUMA_NODE} --membind ${NUMA_NODE} ${dir}/smrt-basecaller "$@"
