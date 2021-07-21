#!/usr/bin/env bash
baz=$1
trc=$2
zmw=$3
f1=$4
f2=$5

outSeq=$(bazviewer --silent -d -n ${zmw} ${baz} | grep Label | cut -d':' -f 2 | xargs | sed 's/[ ",]//g')
expSeq=$(h5dump -d /GroundTruth/Bases -s ${zmw} -c 1 ${trc} | grep "(${zmw})" | cut -d':' -f 2 | sed 's/[ "]//g')

# Need to make sure trace file had a nonzero number of bases for us,
# else we'll loop infinitely below
if [ "${#expSeq}" -eq 0 ]
then
    echo "Bad input trace file, no GroundTruth"
    exit 1
fi

# Replicate out the ground truth, until we have at least as
# many bases as we found in the baz file
repSeq=${expSeq}
while [ "${#repSeq}" -lt "${#outSeq}" ]
do
    repSeq=${repSeq}${expSeq}
done 

# Save to file, just to make examining failed cases easier
echo ${outSeq} > ${f1}
echo ${repSeq} > ${f2}

nwalign ${outSeq} ${repSeq}
