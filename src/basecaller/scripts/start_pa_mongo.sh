# Used to start smrt-basecaller with some small arguments for testing WX integration.
# options:
# BUILD = Release_gcc or Debug_gcc
# GDBSERVER=1  - will run the app under gdbserver. Implies BUILD=Debug_gcc
# GDB=1 - will run the app under gdb. Implies BUILD=Debug_gcc

BUILD=${BUILD:-Release_gcc}
LOGFILTER=${LOGFILTER:-info}
FRAMES=${FRAMES:-1024}
RATE=${RATE:-100}
TRACE_OUTPUT=${TRACE_OUTPUT:-}
INPUT=${INPUT:-alpha}
VSC=${VSC:-0}
NOP=${NOP:-0}
MAXPOPLOOPS=${MAXPOPLOOPS:-10}
TILEPOOLFACTOR=${TILEPOOLFACTOR:-3.0}

if [[ $INPUT == "designer" ]]
then
    INPUT=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
fi

# --numZmwLanes 1 --config layout.lanesPerPool=1 --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024


if [[ $GDBSERVER ]]
then 
#    cmd="/home/UNIXHOME/mlakata/clion-2020.1.1/bin/gdb/linux/bin/gdbserver localhost:2000"
    cmd="/home/UNIXHOME/mlakata/local/gdb-10.1/bin/gdbserver localhost:2000"
    BUILD=Debug_gcc
elif [[ $GDB ]]
then
    cmd="gdb --args"
    BUILD=Debug_gcc
else
    cmd=""
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
nop_option="--nop=${NOP}"

rows=2756
cols=2912
numZmws=$(( $rows * $cols ))
numZmwLanes=$(( numZmws / 64 ))

if false
then
    numZmwLanes=1
    sourceType=TRACE_FILE
else
    sourceType=WX2
    threadClass=$(curl localhost:23602/sras/0/status/other/threadClassName.txt)
    if [[ $threadClass = "SraMongoThread" ]]
    then
      # this is what the FPGA will generate
      fpgaPacketLanes=2
      fpgaFramesPerPacket=$(curl localhost:23602/status/tileDimensions/0)
      fpgaZmwsPerLane=32
      # this is what smrt-basecaller wants
      packetLanes=1
      # framesPerPacket=$(curl localhost:23602/status/tileDimensions/0)
      framesPerPacket=512
      zmwsPerLane=64
    else
      echo ThreadClass $threadClass is not supported yet.
      exit 1
    fi
fi

# maxCallsPerZmw should be increased because packet depth is 512 frames vs 128 frames

tmpjson=$(mktemp)
cat <<HERE > $tmpjson
{
  "algorithm": 
  {
    "pulseAccumConfig": 
    { 
      "maxCallsPerZmw":48 
    }
  },
  "layout" :
  {
      "framesPerChunk": ${framesPerPacket},
      "lanesPerPool" : ${packetLanes},
      "zmwsPerLane": ${zmwsPerLane}
  },
  "source":
  {
    "WX2SourceConfig":
    {
         "dataPath": "HardLoop",
         "maxPopLoops": ${MAXPOPLOOPS},
         "simulatedFrameRate": $RATE,
         "simulatedInputFile": "${INPUT}",
         "sleepDebug": 600,
         "tilePoolFactor" : ${TILEPOOLFACTOR},
         "wxlayout": {
           "framesPerPacket":  ${fpgaFramesPerPacket},
           "lanesPerPacket" : ${fpgaPacketLanes},
           "zmwsPerLane" : ${fpgaZmwsPerLane}
          }
    }
  },
  "traceSaver": 
  {
    "roi": [ [0,0,128,256 ]] // this is not working
    //"roi": [ [0,127],[192,64]] // this is not working either
  }
} 
HERE


cat -n $tmpjson


if [[ $VSC == 0 ]]
then
  cd ../build/x86_64/${BUILD}
else
  cd ../build
fi

set -x
pwd
$cmd ./applications/smrt-basecaller --maxFrames=${FRAMES} --logfilter=${LOGFILTER} --config $tmpjson ${nop_option} ${trc_output}
