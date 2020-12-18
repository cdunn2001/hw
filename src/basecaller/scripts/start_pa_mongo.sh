# Used to start smrt-basecaller with some small arguments for testing WX integration.
# options:
# BUILD = Release_gcc or Debug_gcc
# GDBSERVER=1  - will run the app under gdbserver. Implies BUILD=Debug_gcc
# GDB=1 - will run the app under gdb. Implies BUILD=Debug_gcc

BUILD=${BUILD:-Release_gcc}
LOGFILTER=${LOGFILTER:-info}
FRAMES=${FRAMES:-1024}
RATE=${RATE:-100}
NOP=${NOP:-0}
TRACE_OUTPUT=${TRACE_OUTPUT:-/data/pa/test.trc.h5}
INPUT=${INPUT:-alpha}

if [[ $GDBSERVER ]]
then 
    cmd="/home/UNIXHOME/mlakata/clion-2020.1.1/bin/gdb/linux/bin/gdbserver localhost:2000"
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
        echo "Can't use NOP and TRACE_OUTPUT together"
        exit 1
    fi
    trc_output="--outputtrcfile $TRACE_OUTPUT"
fi

if [[ $NOP  == 1 ]]
then
    nopoption="--nop"
else
    nopoption=""
fi

rows=2756
cols=2912
zmws=$(( rows * cols ))
lanesPerPool=$(( zmws / 64 ))

#      "lanesPerPool": $lanesPerPool,
#      "zmwsPerLane": 64
cat <<HERE > /tmp/config.json
{
  "layout" :
  {
      "framesPerChunk": 512,
      "lanesPerPool" : 1,
      "zmwsPerLane": 64
  },
  "source":
  {
    "sourceType": "WX2",
    "wx2SourceConfig":
    {
         "dataPath": "HardLoop",
         "platform": "Spider",
         "simulatedFrameRate": $RATE
    }
  }
} 
HERE

numZmwLanes=1

cat /tmp/config.json

cd ../build/x86_64/${BUILD}
# sudo $cmd ./applications/smrt-basecaller --outputbazfile=/data/pa/bogus.baz --frames=${FRAMES} --inputfile=constant/123 --logfilter=${LOGFILTER} --config /tmp/config.json ${nop_option} ${trc_output}
set -x
$cmd ./applications/smrt-basecaller --frames=${FRAMES} --numZmwLanes=${numZmwLanes}  --inputfile=${INPUT} --logfilter=${LOGFILTER} --config /tmp/config.json ${nop_option} ${trc_output}
