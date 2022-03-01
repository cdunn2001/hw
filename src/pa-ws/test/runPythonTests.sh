#!/bin/sh

echo "runPythonTests.sh: $0"
pwd

if [[ ! -f e2e_ve/bin/activate ]]
then
  . ./install_python3_ve.sh
fi
. e2e_ve/bin/activate
python3 HttpHelper.py
python3 ProgressHelper.py
python3 SensorSim.py
python3 WxDaemonSim.py
python3 Helpers.py
