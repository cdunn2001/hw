#!/usr/bin/env bash

errors=0

root=`realpath $0`
root=`dirname $root`
if [ -e "$root/pa-ws" ]
then
    # port numbers are defined in PaWsConstants.h
    # please verify that this list is consistent with the header file.
    for port in 23632
    do
        ss -lt | grep ":$port " > /dev/null 2>&1
        if [ $? -eq 0 ]
        then
            let "errors++"
        fi
    done
else
    echo "$root/pa-ws not found, continuing checks..."
    let "errors++"
fi

# Create log directory and set permissions
logdir=/var/log/pacbio/pa-ws
mkdir -p      $logdir
chown pbi:pbi $logdir
chmod 1777    $logdir

echo "$0 errors $errors"
exit $errors
