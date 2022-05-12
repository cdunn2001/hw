This cram test uses experimental configurations that are likely to be changed or removed, once
the real preHQ algorithm is brought on line.  The whole test can probably be removed once
maintenance becomes annoying, as long as something else is dropped in its place

# This first test runs under 2^16 frames as the SpiderMetricBlock that is used for outputting to the BAZ file only supports numFrames as 16-bit. This means aggregating
# beyond 2^16 frames will results in an incorrect number of frames so we limit the current tests to less than that to validate the number of frames are correct from
# a metrics aggregation standpoint. The test below has the HQ-region all starting at the same time so that the preHQ metric block should be empty and all the subsequent
# metric blocks should be aggregated into one as the activity labels are all the same.
  $ BAZFILE=tmp.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":57344, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations  \
  > --outputbazfile ${BAZFILE} --config=prelimHQ.enablePreHQ=true --config=prelimHQ.hqThrottleFraction=1.0 > /dev/null

  $ bazviewer --silent -m -f ${BAZFILE}
  {
  \t"METRICS" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"INTERNAL" : false, (esc)
  \t\t\t"METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 4, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0, (esc)
  \t\t\t\t\t"BASELINE_SD" : 0, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 0, (esc)
  \t\t\t\t\t"PKMID_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 0, (esc)
  \t\t\t\t\t"PKMID_G" : 0, (esc)
  \t\t\t\t\t"PKMID_T" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.1612548828125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.66015625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 26529, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 318, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 352, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1321, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6063, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5071, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4943, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 7852, (esc)
  \t\t\t\t\t"PKMID_G" : 100.0625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.84375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 26588 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
<<<<<<< HEAD
=======
>>>>>>> origin

# Run again but set the hqThrottleFraction=0.25 so start of HQ-region is different resulting in a preHQ metric block with data.
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":57344, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations  \
  > --outputbazfile ${BAZFILE} --config=prelimHQ.enablePreHQ=true --config=prelimHQ.hqThrottleFraction=0.25 > /dev/null

  $ bazviewer --silent -m -i0 ${BAZFILE} 
  {
  \t"METRICS" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"INTERNAL" : false, (esc)
  \t\t\t"METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 4, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0, (esc)
  \t\t\t\t\t"BASELINE_SD" : 0, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 0, (esc)
  \t\t\t\t\t"PKMID_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 0, (esc)
  \t\t\t\t\t"PKMID_G" : 0, (esc)
  \t\t\t\t\t"PKMID_T" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.1612548828125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.66015625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 26529, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 318, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 352, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1321, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6063, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5071, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4943, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 7852, (esc)
  \t\t\t\t\t"PKMID_G" : 100.0625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.84375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 26588 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
<<<<<<< HEAD
=======
>>>>>>> origin

  $ bazviewer --silent -m -i1 ${BAZFILE} 
  {
  \t"METRICS" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"INTERNAL" : false, (esc)
  \t\t\t"METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.022216796875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.15625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 213, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 3, (esc)
  \t\t\t\t\t"NUM_PULSES" : 12, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 227.375, (esc)
  \t\t\t\t\t"PKMID_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 11, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 112, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 70, (esc)
  \t\t\t\t\t"PKMID_G" : 100.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 64.8125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 217 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.1129150390625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.69140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1984, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 99, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 233.125, (esc)
  \t\t\t\t\t"PKMID_C" : 160.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 560, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 547, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 384, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 303, (esc)
  \t\t\t\t\t"PKMID_G" : 103.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 66.6875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1993 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.007442474365234375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.93359375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1866, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 22, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 14, (esc)
  \t\t\t\t\t"NUM_PULSES" : 101, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 233, (esc)
  \t\t\t\t\t"PKMID_C" : 159.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 427, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 337, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 383, (esc)
  \t\t\t\t\t"PKMID_G" : 103.25, (esc)
  \t\t\t\t\t"PKMID_T" : 65.0625, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1873 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.06683349609375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.69140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1972, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 99, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.75, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 556, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 385, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 296, (esc)
  \t\t\t\t\t"PKMID_G" : 102, (esc)
  \t\t\t\t\t"PKMID_T" : 64.6875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1981 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.04132080078125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.95703125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1859, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.25, (esc)
  \t\t\t\t\t"PKMID_C" : 158.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 426, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 326, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 388, (esc)
  \t\t\t\t\t"PKMID_G" : 102.25, (esc)
  \t\t\t\t\t"PKMID_T" : 63.5625, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1864 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.032196044921875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.69140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1971, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 5, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 99, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.875, (esc)
  \t\t\t\t\t"PKMID_C" : 157.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 556, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 384, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 296, (esc)
  \t\t\t\t\t"PKMID_G" : 101.1875, (esc)
  \t\t\t\t\t"PKMID_T" : 63.71875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1980 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.0304107666015625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.94921875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1856, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 6, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.875, (esc)
  \t\t\t\t\t"PKMID_C" : 157.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 425, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 326, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 386, (esc)
  \t\t\t\t\t"PKMID_G" : 101.5625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1861 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.0112762451171875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.7578125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1969, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 7, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.375, (esc)
  \t\t\t\t\t"PKMID_C" : 157.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 296, (esc)
  \t\t\t\t\t"PKMID_G" : 100.3125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.96875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.047882080078125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.95703125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1856, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 8, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.5, (esc)
  \t\t\t\t\t"PKMID_C" : 157.375, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 425, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 326, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 386, (esc)
  \t\t\t\t\t"PKMID_G" : 101.375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.71875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1861 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.0112762451171875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.7578125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1969, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 9, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.125, (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 296, (esc)
  \t\t\t\t\t"PKMID_G" : 100.3125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.96875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.047882080078125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.95703125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1856, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 10, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.375, (esc)
  \t\t\t\t\t"PKMID_C" : 157.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 425, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 326, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 386, (esc)
  \t\t\t\t\t"PKMID_G" : 101.3125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.65625, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1861 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.0112762451171875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.7578125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1969, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 11, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 296, (esc)
  \t\t\t\t\t"PKMID_G" : 100.25, (esc)
  \t\t\t\t\t"PKMID_T" : 62.9375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.047882080078125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.95703125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1856, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 12, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.125, (esc)
  \t\t\t\t\t"PKMID_C" : 156.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 425, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 326, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 386, (esc)
  \t\t\t\t\t"PKMID_G" : 101.125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.53125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1861 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 1, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.0112762451171875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.7578125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1969, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 13, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 18, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 296, (esc)
  \t\t\t\t\t"PKMID_G" : 100.25, (esc)
  \t\t\t\t\t"PKMID_T" : 62.84375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 1, (esc)
  \t\t\t"ZMW_NUMBER" : 1 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin

  $ bazviewer --silent -m -i2 ${BAZFILE} 
  {
  \t"METRICS" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"INTERNAL" : false, (esc)
  \t\t\t"METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.214599609375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.98046875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2061, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 19, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 32, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.625, (esc)
  \t\t\t\t\t"PKMID_C" : 158.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 631, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 304, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 396, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 542, (esc)
  \t\t\t\t\t"PKMID_G" : 103.5, (esc)
  \t\t\t\t\t"PKMID_T" : 65.0625, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2071 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.0712890625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.953125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 21828, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 186, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 222, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 276, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 294, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1014, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.25, (esc)
  \t\t\t\t\t"PKMID_C" : 156.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 5581, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3942, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5292, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5057, (esc)
  \t\t\t\t\t"PKMID_G" : 99.875, (esc)
  \t\t\t\t\t"PKMID_T" : 62.5625, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 21917 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 2, (esc)
  \t\t\t"ZMW_NUMBER" : 2 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin



# Generate complete metrics by disabling preHQ algorithm.
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":16384, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations  \
  > --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer --silent -m -f ${BAZFILE}
  {
  \t"METRICS" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"INTERNAL" : false, (esc)
  \t\t\t"METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.5927734375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.74609375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 215, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0.00082111358642578125, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.00060510635375976562, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.00035643577575683594, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.0013599395751953125, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 2, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 512, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 13, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 6, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 302, (esc)
  \t\t\t\t\t"PKMAX_C" : 201, (esc)
  \t\t\t\t\t"PKMAX_G" : 136, (esc)
  \t\t\t\t\t"PKMAX_T" : 87, (esc)
  \t\t\t\t\t"PKMID_A" : 234.25, (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 60, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 36, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 73, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 20, (esc)
  \t\t\t\t\t"PKMID_G" : 96.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 60.53125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.02734375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 215, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0.646484375 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.1334228515625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.55859375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2017, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.0013475418090820312, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.002140045166015625, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 32, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 4, (esc)
  \t\t\t\t\t"NUM_PULSES" : 102, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 22, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 326, (esc)
  \t\t\t\t\t"PKMAX_C" : 232, (esc)
  \t\t\t\t\t"PKMAX_G" : 153, (esc)
  \t\t\t\t\t"PKMAX_T" : 106, (esc)
  \t\t\t\t\t"PKMID_A" : 231.5, (esc)
  \t\t\t\t\t"PKMID_C" : 161.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 474, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 409, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 293, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 643, (esc)
  \t\t\t\t\t"PKMID_G" : 102.625, (esc)
  \t\t\t\t\t"PKMID_T" : 66.4375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.1328125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2025, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0.70947265625 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.1363525390625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.796875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2053, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.0011491775512695312, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.0017871856689453125, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00097370147705078125, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 21, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 8, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 25, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 318, (esc)
  \t\t\t\t\t"PKMAX_C" : 210, (esc)
  \t\t\t\t\t"PKMAX_G" : 143, (esc)
  \t\t\t\t\t"PKMAX_T" : 104, (esc)
  \t\t\t\t\t"PKMID_A" : 233, (esc)
  \t\t\t\t\t"PKMID_C" : 160.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 452, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 364, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 474, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 565, (esc)
  \t\t\t\t\t"PKMID_G" : 101.875, (esc)
  \t\t\t\t\t"PKMID_T" : 65.5625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.109375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2056, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0.68408203125 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.173583984375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.5390625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2006, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.00064420700073242188, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.001010894775390625, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.0014276504516601562, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 32, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 4, (esc)
  \t\t\t\t\t"NUM_PULSES" : 101, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 22, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 324, (esc)
  \t\t\t\t\t"PKMAX_C" : 229, (esc)
  \t\t\t\t\t"PKMAX_G" : 152, (esc)
  \t\t\t\t\t"PKMAX_T" : 104, (esc)
  \t\t\t\t\t"PKMID_A" : 229.625, (esc)
  \t\t\t\t\t"PKMID_C" : 159.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 472, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 407, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 292, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 637, (esc)
  \t\t\t\t\t"PKMID_G" : 100.625, (esc)
  \t\t\t\t\t"PKMID_T" : 64.6875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.02734375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2012, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0.70849609375 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
<<<<<<< HEAD
=======
>>>>>>> origin
