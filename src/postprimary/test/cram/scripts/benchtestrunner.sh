#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Need three args, got: " $@
fi

data=$1
benchtest-t2b -b basecaller-console-app -b basecaller-console-app --ppa baz2bam --ppa $2 -q production --nworkers 4 --nproc 16 --tmp baz2bam_metrics_benchtest --test --outdir $3 --tmp $3 > $3/benchtest.stdout 2>$3/benchtest.stderr
