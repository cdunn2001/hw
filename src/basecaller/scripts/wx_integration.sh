scriptdir=$(dirname $(realpath $0))

if ( wxinfo | grep Dead )
then
    echo "WX is dead, resetting via wxcontrol"
    sudo wxcontrol -r wxpfw0
fi

pushd ~/git/hw-mongo/scripts
./start_wx_daemon_loopback.sh &
popd

sleep 3

pushd $scriptdir
LOOPBACK=1 RATE=100 TRACE_OUTPUT=/data/pa/mytracefile.trc.h5 NOP=2 FRAMES=5120 ./start_pa_mongo.sh
popd

wait
