#!/usr/bin/env bash

baz=$1
trc=$2
zmw=$3
f1=$4
f2=$5

bazviewer --silent -d -n ${zmw} ${baz} | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g' > ${f1}
h5dump -d /GroundTruth/Bases -s ${zmw} -c 1 ${trc} | grep "(${zmw})" | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' >  ${f2}
outSeq=$(<${f1})
expSeq=$(<${f2})
nwalign ${outSeq} ${expSeq}
