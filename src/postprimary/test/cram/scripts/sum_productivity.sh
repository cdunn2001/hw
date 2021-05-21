#!/bin/bash

source /mnt/software/Modules/current/init/bash
module load xpath
xpath -q -e "/PipeStats/ProdDist/ns:BinCounts/ns:BinCount/text()" $1 | awk '{s+=$1}END{print s}'
