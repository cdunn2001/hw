#!/usr/bin/env bash

errors=0

# If wolverine driver is dead, then reset the whole deal.
if /opt/convey/bin/wxinfo | grep Dead
then
   echo "Resetting Wolverine Driver"
   /opt/convey/bin/wxcontrol -r a0
fi

if /opt/convey/bin/wxinfo | grep Backup_Hix
then
   echo FPGA is in Backup_Hix mode. A firmware reinstallation and power cycle may be necesary.
   exit 1
fi

if [ `cat /proc/sys/kernel/numa_balancing` -ne 0 ]
then
   echo DISABLING NUMA BALANCING, writing 0 to /proc/sys/kernel/numa_balancing
   echo 0 > /proc/sys/kernel/numa_balancing
fi

root=`realpath $0`
root=`dirname $root`
if [ -e "$root/wx-daemon" ]
then
    # port numbers are defined in Sequel/common/pacbio/primary/ipc-config.h
    # please verify that this list is consistent with the header file.
    for port in 23600 23601 23602
    do
        ss -lt | grep ":$port " > /dev/null 2>&1
        if [ $? -eq 0 ]
        then
            let "errors++"
        fi
    done
else
    echo "$root/wx-daemon not found, continuing checks..."
    let "errors++"
fi

# Create log directory and set permissions
logdir=/var/log/pacbio/wx-daemon
mkdir -p      $logdir
chown pbi:pbi $logdir
chmod 1777    $logdir

echo "$0 errors $errors"
exit $errors
