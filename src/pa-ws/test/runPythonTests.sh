#!/bin/sh

echo "runPythonTests.sh: $0"
pwd

if [[ ! -f e2e_ve/bin/activate ]]
then
  . ./install_python3_ve.sh
fi
. e2e_ve/bin/activate

netStatus=0
runtest() {
  python3 $1
  netStatus=$(( netStatus+$? ))
}

runtest Helpers.py
runtest HttpHelper.py
runtest KestrelRT.py
runtest ProgressHelper.py
runtest SensorSim.py
runtest WxDaemonSim.py

echo "Net exit status: $netStatus"
exit $netStatus
