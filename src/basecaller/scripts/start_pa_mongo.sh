# Used to start smrt-basecaller with some small arguments for testing WX integration.
# options:
# BUILD = Release or Debug or RelWithDebInfo
# GDBSERVER=1  - will run the app under gdbserver. Implies BUILD=Debug
# GDB=1 - will run the app under gdb. Implies BUILD=Debug

BUILD=${BUILD:-Release}
LOGFILTER=${LOGFILTER:-info}
LOGOUTPUT=${LOGOUTPUT:-/var/log/pacbio/pa-smrtbasecaller/pa-smrtbasecaller}
FRAMES=${FRAMES:-1024}
RATE=${RATE:-100}
TRACE_OUTPUT=${TRACE_OUTPUT:-}
BAZ_OUTPUT=${BAZ_OUTPUT:-}
INPUT=${INPUT:-alpha}
VSC=${VSC:-0}
NOP=${NOP:-0}
MAXPOPLOOPS=${MAXPOPLOOPS:-10}
TILEPOOLFACTOR=${TILEPOOLFACTOR:-3.0}
LOOPBACK=${LOOPBACK:-false}
SRA_INDEX=${SRA_INDEX:-0}  # 0 to 3
ROI=${ROI:-[[0,0,64,256]]}  # don't use spaces. This can also be a filename to a JSON file that contains the ROI.
PRIMERSCALER=${PRIMERSCALER:-2.0}

# append this directory to the PATH
scriptdir=$(dirname $(realpath $0))

if [[ $INPUT == "designer" ]]
then
    INPUT=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
fi

# --numZmwLanes 1 --config layout.lanesPerPool=1 --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024


if [[ $GDBSERVER ]]
then 
#    cmd="/home/UNIXHOME/mlakata/clion-2020.1.1/bin/gdb/linux/bin/gdbserver localhost:2000"
    cmd="/home/UNIXHOME/mlakata/local/gdb-10.1/bin/gdbserver localhost:2000"
    BUILD=Debug
elif [[ $GDB ]]
then
    cmd="gdb --args"
    BUILD=Debug
elif [[ $ECHO ]]
then
    cmd="echo "
else
    cmd=""
    GPU_ID=0
    NUMA_NODE=0
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    cmd="numactl --cpubind ${NUMA_NODE} --membind ${NUMA_NODE}"
fi

if [[ $TRACE_OUTPUT != "" ]]
then
    if [[ $NOP == 1 ]]
    then  
       echo "NOP=1 can't be used with TRACE_OUTPUT = $TRACE_OUTPUT."
       exit 1
    fi
    trc_output="--outputtrcfile $TRACE_OUTPUT"
    rm -f $TRACE_OUTPUT
fi
if [[ $BAZ_OUTPUT != "" ]]
then
  baz_output="--outputbazfile $BAZ_OUTPUT"
else
  baz_output=""
fi
nop_option="--nop=${NOP}"

if false
then
    sourceType=TRACE_FILE
else
    sourceType=WX2
fi

# maxCallsPerZmw should be increased because packet depth is 512 frames vs 128 frames

tmpjson=$(mktemp)
acqconfig=$(mktemp)
cat <<HERE > $tmpjson
{
  "algorithm": 
  {
    "pulseAccumConfig": 
    { 
      "maxCallsPerZmw":48 
    }
  },
  "source":
  {
    "WXIPCDataSourceConfig":
    {
         "sraIndex": $SRA_INDEX,
         "dataPath": "HardLoop",
         "maxPopLoops": ${MAXPOPLOOPS},
         "simulatedFrameRate": $RATE,
         "simulatedInputFile": "${INPUT}",
         "sleepDebug": 600,
         "tilePoolFactor" : ${TILEPOOLFACTOR},
         "loopback": ${LOOPBACK},
         "primerScaler": ${PRIMERSCALER}
    }
  },
  "system":
  {
    "maxPermGpuDataMB":17000
  }
} 
HERE


cat <<HERE > $acqconfig
{
  "source":
  {
    "WXIPCDataSourceConfig":
    {
      "acqConfig" :
      {
        "refSnr":  12.0,
        "C" : {
          "baseLabel": "C",
          "relAmplitude": 1.0,
          "excessNoiseCV": 0.1,
          "interPulseDistance": 0.07,
          "pulseWidth": 0.209,
          "pw2SlowStepRatio": 3.2,
          "ipd2SlowStepRatio": 0
        },
        "A" : {
          "baseLabel": "A",
          "relAmplitude": 0.67,
          "excessNoiseCV": 0.1,
          "interPulseDistance": 0.08,
          "pulseWidth": 0.166,
          "pw2SlowStepRatio": 3.2,
          "ipd2SlowStepRatio": 0
        },
        "T" : {
          "baseLabel": "T",
          "relAmplitude": 0.445,
          "excessNoiseCV": 0.1,
          "interPulseDistance": 0.08,
          "pulseWidth": 0.163,
          "pw2SlowStepRatio": 3.2,
          "ipd2SlowStepRatio": 0
        },
        "G" : {
          "baseLabel": "G",
          "relAmplitude": 0.26,
          "excessNoiseCV": 0.1,
          "interPulseDistance": 0.07,
          "pulseWidth": 0.193,
          "pw2SlowStepRatio": 3.2,
          "ipd2SlowStepRatio": 0
        }
      }
    }
  }
}
HERE

cat -n $tmpjson
cat -n $acqconfig

# prepend to PATH
if [[ $scriptdir == /opt/pacbio* ]]
then
  # no change to PATH
  true
elif [[ $VSC == 1 ]]
then
  # Visual studio build dir
  export PATH=$scriptdir/build_vsc:${PATH}
else
  # normal build dir
  export PATH=$scriptdir/../../../build/basecaller/gcc/x86_64/${BUILD}/applications:${PATH}
  echo PATH is $PATH
fi
if [[ $LOGOUTPUT != "" && $LOGOUTPUT != "none" ]]
then
    logoutput="--logoutput $LOGOUTPUT"
else
    logoutput=""
fi
if [[ -f $ROI ]]
then
  if ! grep traceSaver $ROI
  then
     echo "The $ROI file needs to be written in a JSON object that looks like"
     echo "  { \"traceSaver\": { \"roi\": "
     cat   $ROI 
     echo "  } }"
     exit 1
  fi
  roi_spec="--config $ROI"
else
  roi_spec="--config traceSaver.roi=$ROI"
fi


echo PATH = $PATH

set -x
pwd
$cmd smrt-basecaller --maxFrames=${FRAMES} --logfilter=${LOGFILTER} --config $tmpjson --config $acqconfig ${nop_option} ${trc_output} ${logoutput} ${roi_spec} ${baz_output}

