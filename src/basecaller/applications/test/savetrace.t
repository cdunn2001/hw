  $ TRCOUT=${CRAMTMP}/out.trc.h5
  $ TRCIN=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCIN'"}' \
  > --nop=2 --config layout.lanesPerPool=1 --outputtrcfile ${TRCOUT} --config=traceSaver.roi=[[0,127],[192,64]] > /dev/null

# The original trace had 256 ZMW, but we only selected 192 via the ROI
  $ h5ls ${TRCOUT}/TraceData
  HoleNumber               Dataset {192}
  HoleType                 Dataset {192}
  HoleXY                   Dataset {192, 2}
  Traces                   Dataset {192, 1, 8192}

  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCOUT} > Zmw0.out
  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCIN} > Zmw0.in

# ZMW 0 should look identical in the two trace files, minus the overall
# dataset size
  $ diff -u Zmw0.in Zmw0.out
  --- Zmw0.in* (glob)
  +++ Zmw0.out* (glob)
  @@ -1,7 +1,7 @@
  -HDF5 "/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5" {
  +HDF5 "*/out.trc.h5" { (glob)
   DATASET "TraceData/Traces" {
      DATATYPE  H5T_STD_I16LE
  -   DATASPACE  SIMPLE { ( 256, 1, 8192 ) / ( 256, 1, 8192 ) }
  +   DATASPACE  SIMPLE { ( 192, 1, 8192 ) / ( 192, 1, 8192 ) }
      SUBSET {
         START ( 0, 0, 0 );
         STRIDE ( 1, 1, 1 );
  [1]

# ZMW 64 also should look identical in the two trace files
  $ h5dump -d TraceData/Traces -s "64,0,0" -c "1,1,8192" ${TRCOUT} > Zmw64.out
  $ h5dump -d TraceData/Traces -s "64,0,0" -c "1,1,8192" ${TRCIN} > Zmw64.in

  $ diff -u Zmw64.in Zmw64.out
  --- Zmw64.in* (glob)
  +++ Zmw64.out* (glob)
  @@ -1,7 +1,7 @@
  -HDF5 "/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5" {
  +HDF5 "*/out.trc.h5" { (glob)
   DATASET "TraceData/Traces" {
      DATATYPE  H5T_STD_I16LE
  -   DATASPACE  SIMPLE { ( 256, 1, 8192 ) / ( 256, 1, 8192 ) }
  +   DATASPACE  SIMPLE { ( 192, 1, 8192 ) / ( 192, 1, 8192 ) }
      SUBSET {
         START ( 64, 0, 0 );
         STRIDE ( 1, 1, 1 );
  [1]

# filter out the zmw number from the data, since now we're doing different zmw
  $ h5dump -d TraceData/Traces -s "128,0,0" -c "1,1,8192" ${TRCOUT} | sed 's/128,0//' > Zmw128.out
  $ h5dump -d TraceData/Traces -s "192,0,0" -c "1,1,8192" ${TRCIN} | sed 's/192,0//' > Zmw192.in

# Now due to how the ROI was specified, ZMW 128 in the output should have the same data
# as 192 in the input
  $ diff -u Zmw192.in Zmw128.out
  --- Zmw192.in* (glob)
  +++ Zmw128.out* (glob)
  @@ -1,9 +1,9 @@
  -HDF5 "/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5" {
  +HDF5 "*/out.trc.h5" { (glob)
   DATASET "TraceData/Traces" {
      DATATYPE  H5T_STD_I16LE
  -   DATASPACE  SIMPLE { ( 256, 1, 8192 ) / ( 256, 1, 8192 ) }
  +   DATASPACE  SIMPLE { ( 192, 1, 8192 ) / ( 192, 1, 8192 ) }
      SUBSET {
  -      START ( 192, 0, 0 );
  +      START ( 128, 0, 0 );
         STRIDE ( 1, 1, 1 );
         COUNT ( 1, 1, 8192 );
         BLOCK ( 1, 1, 1 );
  [1]

# Read the input trace file as 8 bit even though it really is 16.
# This will force saturation for values out of bounds
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCIN'", "inputType":"UINT8"}' \
  > --nop=2 --config layout.lanesPerPool=1 --outputtrcfile ${TRCOUT} --config=traceSaver.roi=[[0,127],[192]] > /dev/null

  $ h5ls ${TRCOUT}/TraceData
  HoleNumber               Dataset {192}
  HoleType                 Dataset {192}
  HoleXY                   Dataset {192, 2}
  Traces                   Dataset {192, 1, 8192}

  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCOUT} | head -n 20 > Zmw0Trunc.out
  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCIN} | head -n 20 > Zmw0NoTrunc.in

# All apparent difference below are from the saturation, i.e. they are less than 0
# or greater than 255 in the input file
  $ diff -u Zmw0Trunc.out Zmw0NoTrunc.in
  --- Zmw0Trunc.out* (glob)
  +++ Zmw0NoTrunc.in* (glob)
  @@ -1,20 +1,20 @@
  -HDF5 "*/out.trc.h5" { (glob)
  +HDF5 "/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5" {
   DATASET "TraceData/Traces" {
  -   DATATYPE  H5T_STD_U8LE
  -   DATASPACE  SIMPLE { ( 192, 1, 8192 ) / ( 192, 1, 8192 ) }
  +   DATATYPE  H5T_STD_I16LE
  +   DATASPACE  SIMPLE { ( 256, 1, 8192 ) / ( 256, 1, 8192 ) }
      SUBSET {
         START ( 0, 0, 0 );
         STRIDE ( 1, 1, 1 );
         COUNT ( 1, 1, 8192 );
         BLOCK ( 1, 1, 1 );
         DATA {
  -      (0,0,0): 0, 7, 1, 7, 0, 0, 2, 0, 0, 2, 0, 1, 2, 0, 0, 1, 2, 125, 255,
  -      (0,0,19): 116, 0, 4, 6, 2, 0, 0, 0, 0, 10, 175, 255, 255, 203, 206,
  -      (0,0,34): 191, 255, 242, 209, 232, 236, 208, 243, 235, 237, 229, 238,
  -      (0,0,46): 217, 255, 255, 247, 255, 239, 185, 200, 209, 228, 39, 0, 7,
  -      (0,0,59): 16, 0, 4, 14, 4, 11, 5, 5, 8, 4, 0, 0, 4, 0, 0, 153, 104,
  -      (0,0,76): 178, 153, 122, 2, 2, 2, 126, 240, 243, 244, 255, 227, 231,
  -      (0,0,89): 229, 222, 249, 245, 242, 174, 208, 238, 255, 213, 186, 255,
  -      (0,0,101): 184, 255, 250, 150, 255, 244, 226, 221, 255, 243, 178, 203,
  -      (0,0,113): 255, 101, 1, 0, 0, 6, 1, 0, 5, 0, 0, 3, 0, 6, 12, 18, 4, 0,
  -      (0,0,131): 3, 0, 11, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 6, 6, 0, 3, 3, 11,
  +      (0,0,0): -2, 7, 1, 7, -3, 0, 2, 0, -4, 2, -2, 1, 2, -2, -11, 1, 2, 125,
  +      (0,0,18): 271, 116, -11, 4, 6, 2, -6, 0, -1, -5, 10, 175, 261, 278,
  +      (0,0,32): 203, 206, 191, 261, 242, 209, 232, 236, 208, 243, 235, 237,
  +      (0,0,44): 229, 238, 217, 256, 257, 247, 270, 239, 185, 200, 209, 228,
  +      (0,0,56): 39, -5, 7, 16, -2, 4, 14, 4, 11, 5, 5, 8, 4, -2, -4, 4, -5,
  +      (0,0,73): 0, 153, 104, 178, 153, 122, 2, 2, 2, 126, 240, 243, 244, 273,
  +      (0,0,87): 227, 231, 229, 222, 249, 245, 242, 174, 208, 238, 272, 213,
  +      (0,0,99): 186, 274, 184, 262, 250, 150, 301, 244, 226, 221, 277, 243,
  +      (0,0,111): 178, 203, 258, 101, 1, -1, -5, 6, 1, -6, 5, -1, 0, 3, -3, 6,
  +      (0,0,127): 12, 18, 4, -1, 3, -3, 11, -3, -3, 2, 2, -1, -6, -1, -2, -3,
  [1]

# This time read in 8 bit traces, but output a 16 bit trace file.  We're still
# dealing with a true 16 bit input so there are saturated values, but this run
# should now look identical to the last, save for the data type specified in
# the output trace file
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCIN'", "inputType":"UINT8"}' \
  > --nop=2 --config layout.lanesPerPool=1 --outputtrcfile ${TRCOUT} --config=traceSaver='{ "roi" : [[0,127],[192]], "outFormat":"INT16"}' > /dev/null

  $ h5ls ${TRCOUT}/TraceData
  HoleNumber               Dataset {192}
  HoleType                 Dataset {192}
  HoleXY                   Dataset {192, 2}
  Traces                   Dataset {192, 1, 8192}

  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCOUT} | head -n 20 > Zmw0TruncAndExpand.out

  $ diff -u Zmw0Trunc.out Zmw0TruncAndExpand.out
  --- Zmw0Trunc.out* (glob)
  +++ Zmw0TruncAndExpand.out* (glob)
  @@ -1,6 +1,6 @@
   HDF5 "*/out.trc.h5" { (glob)
   DATASET "TraceData/Traces" {
  -   DATATYPE  H5T_STD_U8LE
  +   DATATYPE  H5T_STD_I16LE
      DATASPACE  SIMPLE { ( 192, 1, 8192 ) / ( 192, 1, 8192 ) }
      SUBSET {
         START ( 0, 0, 0 );
  [1]
