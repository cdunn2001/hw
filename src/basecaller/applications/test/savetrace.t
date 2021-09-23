  $ TRCOUT=${CRAMTMP}/out.trc.h5
  $ TRCIN=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCIN'"}' \
  > --nop=2 --config layout.lanesPerPool=1 --outputtrcfile ${TRCOUT} --config=traceSaver.roi=[[0,127],[192,64]] > /dev/null

  $ h5ls ${TRCOUT}/TraceData
  HoleNumber               Dataset {192}
  HoleType                 Dataset {192}
  HoleXY                   Dataset {192, 2}
  Traces                   Dataset {192, 1, 8192}

  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCOUT} > Zmw0.out
  $ h5dump -d TraceData/Traces -s "0,0,0" -c "1,1,8192" ${TRCIN} > Zmw0.in

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
