#!/bin/bash
CARD=$(/opt/micron/bin/wxinfo -v | grep "Wolverine Board Info:" | awk '{print $NF}')
FPGA=$(/opt/micron/bin/wxinfo -v | grep "Architecture:" | awk '{print $NF}')
if [ "$FPGA" == "WX2_VU7P" ]
then
	echo "Enabling Shadow Page Table for WX2 card."
	/opt/micron/sbin/wxspt ${CARD} enable > /dev/null
	/opt/micron/sbin/wxspt ${CARD} 32 > /dev/null
	/opt/micron/sbin/wxspt ${CARD} list
else
	echo "Shadow Page Table not supported for ${FPGA}."
fi