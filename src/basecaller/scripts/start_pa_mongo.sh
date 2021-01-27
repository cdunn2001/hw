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
VSC=${VSC:-1}
NOP=${NOP:-0}


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
fi
nop_option="--nop=${NOP}"

rows=2756
cols=2912
numZmws=$(( $rows * $cols ))
numZmwLanes=$(( numZmws / 64 ))

# maxCallsPerZmw should be increased because packet depth is 512 frames vs 128 frames

cat <<HERE > /tmp/config.json
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
      "framesPerChunk": 512,
      "lanesPerPool" : 4096,
      "zmwsPerLane": 64
  },
  "source":
  {
    "sourceType": "WX2",
    "wx2SourceConfig":
    {
         "dataPath": "HardLoop",
         "platform": "Sequel2Lvl1",
         "simulatedFrameRate": $RATE,
         "sleepDebug": 600,
         "wxlayout": {
           "lanesPerPacket" : 1024,
           "framesPerPacket":  512,
           "zmwsPerLane" : 32
          }
    }
  },
  "traceROI": {
    "roi": [ [ 0,0, 1, 64 ], [0,64,1,64], [2, 64, 1, 64] ]
  }
} 
HERE


cat /tmp/config.json

if [[ $VSC == 0 ]]
then
  cd ../build/x86_64/${BUILD}
else
  cd ../build
fi

# sudo $cmd ./applications/smrt-basecaller --outputbazfile=/data/pa/bogus.baz --frames=${FRAMES} --inputfile=constant/123 --logfilter=${LOGFILTER} --config /tmp/config.json ${nop_option} ${trc_output}
set -x
$cmd ./applications/smrt-basecaller --frames=${FRAMES} --numZmwLanes=${numZmwLanes}  --inputfile=${INPUT} --logfilter=${LOGFILTER} --config /tmp/config.json ${nop_option} ${trc_output}
