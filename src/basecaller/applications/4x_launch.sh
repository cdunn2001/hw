set -e

DELAY=${DELAY:-0}
INPUT=${INPUT:-/pbi/dept/primary/sim/mongo/test3_mongo_SNR-40.trc.h5}
OUTPUT_LOC=${OUTPUT_LOC:-/data/pa}
BAZ_NAME=${BAZ_NAME:-tmp.baz}
FRAMES=${FRAMES:-100000}
LANES_PER_CHIP=${LANES_PER_CHIP:-393216}
PERM_GPU_MB=${PERM_GPU_MB:-9500}
BASECALLER_CONCURRENCY=${BASECALLER_CONCURRENCY:-3}
DME_FULL_ITERATE=${DME_FULL_ITERATE:-1}

dir=$(dirname $0)

BAZ_ARG=${OUTPUT_LOC}/${BAZ_NAME%.baz}

NUM_A100=$(nvidia-smi | grep -c A100)
if [[ ${NUM_A100} -ne 2 ]]
then
    echo "Expecting a system with 2 A100 cards, found ${NUM_A100}"
    exit 1
fi

ps aux | grep nvidia-cuda-mps-control | grep -v grep > /dev/null
if [[ $? -ne 0 ]]
then
    echo "MPS dameon is not running!"
    echo "Suggested command to run as root: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60 /usr/bin/nvidia-cuda-mps-control -d"
    exit 1
fi

THREAD_PCNT=$(echo get_default_active_thread_percentage | /usr/bin/nvidia-cuda-mps-control)

if [[ $(echo ${THREAD_PCNT} != 60 | bc) == 1 ]]
then
    echo "MPS daemon is not set up with the expected resource provisioning.  Set active thread percentage to 60 or disable this check"
    exit 1
fi

MAIN_ARGS="--cache --numZmwLanes ${LANES_PER_CHIP} --frames=${FRAMES}  --inputfile ${INPUT} \
           --config=algorithm.dmeConfig.IterateToLimit=${DME_FULL_ITERATE} \
           --config=system.maxPermGpuDataMB=${PERM_GPU_MB} \
           --config=system.basecallerConcurrency=${BASECALLER_CONCURRENCY} \
           $@"

NUMA_NODE=0 ${dir}/numa_launch.sh ${MAIN_ARGS}  --outputbazfile=${BAZ_ARG}0.baz > t2b0.log &
sleep ${DELAY}

NUMA_NODE=0 ${dir}/numa_launch.sh ${MAIN_ARGS}  --outputbazfile=${BAZ_ARG}1.baz > t2b1.log &
sleep ${DELAY}

NUMA_NODE=1 ${dir}/numa_launch.sh ${MAIN_ARGS}  --outputbazfile=${BAZ_ARG}2.baz > t2b2.log &
sleep ${DELAY}

NUMA_NODE=1 ${dir}/numa_launch.sh ${MAIN_ARGS}  --outputbazfile=${BAZ_ARG}3.baz > t2b3.log &

wait

trap "kill -9 -p 0" SIGINT

