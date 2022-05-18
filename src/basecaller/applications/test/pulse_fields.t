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
  \t\t\t\t\t"PulseWidth" : 14, (esc)
  \t\t\t\t\t"StartFrame" : 273 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 8, (esc)
  \t\t\t\t\t"PulseWidth" : 4, (esc)
  \t\t\t\t\t"StartFrame" : 287 (esc)
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
  \t\t\t\t\t"PulseWidth" : 15, (esc)
  \t\t\t\t\t"StartFrame" : 349 (esc)
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
  \t\t\t\t\t"Pkmax" : 270, (esc)
  \t\t\t\t\t"Pkmean" : 169, (esc)
  \t\t\t\t\t"Pkmid" : 270, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 3, (esc)
  \t\t\t\t\t"StartFrame" : 17 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 1, (esc)
  \t\t\t\t\t"Pkmax" : 276, (esc)
  \t\t\t\t\t"Pkmean" : 221.10000610351562, (esc)
  \t\t\t\t\t"Pkmid" : 230, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 28, (esc)
  \t\t\t\t\t"StartFrame" : 29 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "C", (esc)
  \t\t\t\t\t"POS" : 2, (esc)
  \t\t\t\t\t"Pkmax" : 176, (esc)
  \t\t\t\t\t"Pkmean" : 140.39999389648438, (esc)
  \t\t\t\t\t"Pkmid" : 143.30000305175781, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 5, (esc)
  \t\t\t\t\t"StartFrame" : 74 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 3, (esc)
  \t\t\t\t\t"Pkmax" : 300, (esc)
  \t\t\t\t\t"Pkmean" : 223.69999694824219, (esc)
  \t\t\t\t\t"Pkmid" : 230.89999389648438, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 33, (esc)
  \t\t\t\t\t"StartFrame" : 82 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "A", (esc)
  \t\t\t\t\t"POS" : 4, (esc)
  \t\t\t\t\t"Pkmax" : 274, (esc)
  \t\t\t\t\t"Pkmean" : 187.5, (esc)
  \t\t\t\t\t"Pkmid" : 249, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 4, (esc)
  \t\t\t\t\t"StartFrame" : 158 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "T", (esc)
  \t\t\t\t\t"POS" : 5, (esc)
  \t\t\t\t\t"Pkmax" : 63, (esc)
  \t\t\t\t\t"Pkmean" : 47.299999237060547, (esc)
  \t\t\t\t\t"Pkmid" : 54, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 7, (esc)
  \t\t\t\t\t"StartFrame" : 177 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 6, (esc)
  \t\t\t\t\t"Pkmax" : 130, (esc)
  \t\t\t\t\t"Pkmean" : 91.099998474121094, (esc)
  \t\t\t\t\t"Pkmid" : 93.199996948242188, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 41, (esc)
  \t\t\t\t\t"StartFrame" : 194 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 7, (esc)
  \t\t\t\t\t"Pkmax" : 122, (esc)
  \t\t\t\t\t"Pkmean" : 88.900001525878906, (esc)
  \t\t\t\t\t"Pkmid" : 96.199996948242188, (esc)
  \t\t\t\t\t"Pkvar" : 1e+9999, (esc)
  \t\t\t\t\t"PulseWidth" : 14, (esc)
  \t\t\t\t\t"StartFrame" : 273 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"IsBase" : true, (esc)
  \t\t\t\t\t"Label" : "G", (esc)
  \t\t\t\t\t"POS" : 8, (esc)
  \t\t\t\t\t"Pkmax" : 92, (esc)
  \t\t\t\t\t"Pkmean" : 67.5, (esc)
