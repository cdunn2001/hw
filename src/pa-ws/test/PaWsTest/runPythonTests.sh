#!/bin/sh

echo "runPythonTests.sh: $0"
pwd

if [[ ! -f ../e2e_ve/bin/activate ]]
then
    pushd ..
    . ./install_python3_ve.sh
    popd
fi
. ../e2e_ve/bin/activate

exec pytest *.py
