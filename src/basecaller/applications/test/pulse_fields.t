  $ BAZFILE=${CRAMTMP}/test4.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations \
  > --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer -d ${BAZFILE} -i 0 | grep -v info | head -n 100
  Finding Metadata locations
  Read 3 SUPER_CHUNK_META
  {
  \t"STITCHED" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"DATA" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 0, (esc)
  \t\t\t\t\t"PulseWidth" : 3, (esc)
  \t\t\t\t\t"StartFrame" : 17 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 1, (esc)
  \t\t\t\t\t"PulseWidth" : 28, (esc)
  \t\t\t\t\t"StartFrame" : 29 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "C", (esc)
  \t\t\t\t\t"POS" : 2, (esc)
  \t\t\t\t\t"PulseWidth" : 5, (esc)
  \t\t\t\t\t"StartFrame" : 74 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 3, (esc)
  \t\t\t\t\t"PulseWidth" : 33, (esc)
  \t\t\t\t\t"StartFrame" : 82 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 4, (esc)
  \t\t\t\t\t"PulseWidth" : 4, (esc)
  \t\t\t\t\t"StartFrame" : 158 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "T", (esc)
  \t\t\t\t\t"POS" : 5, (esc)
  \t\t\t\t\t"PulseWidth" : 7, (esc)
  \t\t\t\t\t"StartFrame" : 177 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 6, (esc)
  \t\t\t\t\t"PulseWidth" : 41, (esc)
  \t\t\t\t\t"StartFrame" : 194 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 7, (esc)
  \t\t\t\t\t"PulseWidth" : 15, (esc)
  \t\t\t\t\t"StartFrame" : 273 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 8, (esc)
  \t\t\t\t\t"PulseWidth" : 3, (esc)
  \t\t\t\t\t"StartFrame" : 288 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 9, (esc)
  \t\t\t\t\t"PulseWidth" : 22, (esc)
  \t\t\t\t\t"StartFrame" : 299 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "T", (esc)
  \t\t\t\t\t"POS" : 10, (esc)
  \t\t\t\t\t"PulseWidth" : 17, (esc)
  \t\t\t\t\t"StartFrame" : 348 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "C", (esc)
  \t\t\t\t\t"POS" : 11, (esc)
  \t\t\t\t\t"PulseWidth" : 8, (esc)
  \t\t\t\t\t"StartFrame" : 404 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "C", (esc)
  \t\t\t\t\t"POS" : 12, (esc)
  \t\t\t\t\t"PulseWidth" : 29, (esc)
  \t\t\t\t\t"StartFrame" : 446 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)

  $ BAZFILE=${CRAMTMP}/test4_internal.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations \
  > --outputbazfile ${BAZFILE} --config=internalMode=true> /dev/null

  $ bazviewer -d ${BAZFILE} -i 0 | grep -v info | head -n 100
  Finding Metadata locations
  Read 3 SUPER_CHUNK_META
  {
  \t"STITCHED" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"DATA" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 0, (esc)
  \t\t\t\t\t"Pkmax" : 272, (esc)
  \t\t\t\t\t"Pkmean" : 171.30000305175781, (esc)
  \t\t\t\t\t"Pkmid" : 272, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 3, (esc)
  \t\t\t\t\t"StartFrame" : 17 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 1, (esc)
  \t\t\t\t\t"Pkmax" : 278, (esc)
  \t\t\t\t\t"Pkmean" : 223.10000610351562, (esc)
  \t\t\t\t\t"Pkmid" : 232, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 28, (esc)
  \t\t\t\t\t"StartFrame" : 29 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "C", (esc)
  \t\t\t\t\t"POS" : 2, (esc)
  \t\t\t\t\t"Pkmax" : 178, (esc)
  \t\t\t\t\t"Pkmean" : 142.39999389648438, (esc)
  \t\t\t\t\t"Pkmid" : 145.30000305175781, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 5, (esc)
  \t\t\t\t\t"StartFrame" : 74 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 3, (esc)
  \t\t\t\t\t"Pkmax" : 302, (esc)
  \t\t\t\t\t"Pkmean" : 225.69999694824219, (esc)
  \t\t\t\t\t"Pkmid" : 232.89999389648438, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 33, (esc)
  \t\t\t\t\t"StartFrame" : 82 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 4, (esc)
  \t\t\t\t\t"Pkmax" : 276, (esc)
  \t\t\t\t\t"Pkmean" : 189.5, (esc)
  \t\t\t\t\t"Pkmid" : 251, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 4, (esc)
  \t\t\t\t\t"StartFrame" : 158 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "T", (esc)
  \t\t\t\t\t"POS" : 5, (esc)
  \t\t\t\t\t"Pkmax" : 66, (esc)
  \t\t\t\t\t"Pkmean" : 49.700000762939453, (esc)
  \t\t\t\t\t"Pkmid" : 56.400001525878906, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 7, (esc)
  \t\t\t\t\t"StartFrame" : 177 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 6, (esc)
  \t\t\t\t\t"Pkmax" : 132, (esc)
  \t\t\t\t\t"Pkmean" : 93.199996948242188, (esc)
  \t\t\t\t\t"Pkmid" : 95.199996948242188, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 41, (esc)
  \t\t\t\t\t"StartFrame" : 194 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 7, (esc)
  \t\t\t\t\t"Pkmax" : 124, (esc)
  \t\t\t\t\t"Pkmean" : 85.699996948242188, (esc)
  \t\t\t\t\t"Pkmid" : 96.800003051757812, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 15, (esc)
  \t\t\t\t\t"StartFrame" : 273 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 8, (esc)
  \t\t\t\t\t"Pkmax" : 94, (esc)
  \t\t\t\t\t"Pkmean" : 88, (esc)
