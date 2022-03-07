scriptdir=$(dirname $(realpath $0))

WX_DAEMON_WORKSPACE=${WX_DAEMON_WORKSPACE:-${scriptdir}/../../../hw-mongo}
if [[ ! -d ${WX_DAEMON_WORKSPACE} ]]
then
    echo "This script requires ${WX_DAEMON_WORKSPACE} exists. Try setting the WX_DAEMON_WORKSPACE envvar to point to git/hw-mongo"
    exit 1
fi

pushd ${WX_DAEMON_WORKSPACE}/scripts
./start_wx_daemon_loopback.sh &
popd

sleep 3

pushd $scriptdir
LOOPBACK=1 RATE=100 TRACE_OUTPUT=/data/pa/mytracefile.trc.h5 NOP=2 FRAMES=5120 ./start_pa_mongo.sh
popd

# This shuts down wx-daemon safely.
curl -X POST http://localhost:23602/shutdown

echo "Waiting for wx-daemon to shutdown"
wait
