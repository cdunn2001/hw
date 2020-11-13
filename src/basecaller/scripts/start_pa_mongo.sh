# Used to start smrt-basecaller with some small arguments for testing WX integration.
# options:
# BUILD = Release_gcc or Debug_gcc
# GDBSERVER=1  - will run the app under gdbserver. Implies BUILD=Debug_gcc
# GDB=1 - will run the app under gdb. Implies BUILD=Debug_gcc

BUILD=${BUILD:-Release_gcc}
LOGFILTER=${LOGFILTER:-info}

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

cd ../build/x86_64/${BUILD}
sudo $cmd ./applications/smrt-basecaller --config source.sourceType=WX2 --outputbazfile=/data/pa/bogus.baz --frames=1024 --inputfile=constant --config layout.framesPerChunk=512 --config source.wx2SourceConfig.dataPath=HardLoop --config source.wx2SourceConfig.platform=Spider --logfiler=${LOGFILTER}
