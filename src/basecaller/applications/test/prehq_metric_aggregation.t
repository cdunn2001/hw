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
  \t\t\t\t\t"BASELINE_MEAN" : 12.6015625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 8.9765625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 26586, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 318, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 372, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1321, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6063, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5071, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4943, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 7869, (esc)
  \t\t\t\t\t"PKMID_G" : 100.0625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.8125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 26588 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }

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
  \t\t\t\t\t"BASELINE_MEAN" : 12.6015625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 8.9765625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 26586, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 318, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 372, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1321, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6063, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5071, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4943, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 7869, (esc)
  \t\t\t\t\t"PKMID_G" : 100.0625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.8125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 26588 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }

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
  \t\t\t\t\t"BASELINE_MEAN" : 0.9775390625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.15625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 217, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 6, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 3.3515625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.94140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1989, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 99, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 233.125, (esc)
  \t\t\t\t\t"PKMID_C" : 160.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 560, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 547, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 384, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 304, (esc)
  \t\t\t\t\t"PKMID_G" : 103.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 66.6875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1993 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 6.18359375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 6.60546875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1871, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 8.59375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 6.9140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1977, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 99, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.75, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 556, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 385, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 297, (esc)
  \t\t\t\t\t"PKMID_G" : 102, (esc)
  \t\t\t\t\t"PKMID_T" : 64.75, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1981 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 10.15625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.6875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1864, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 11.1875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 8.5078125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1976, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 5, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 99, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.875, (esc)
  \t\t\t\t\t"PKMID_C" : 157.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 556, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 384, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 297, (esc)
  \t\t\t\t\t"PKMID_G" : 101.1875, (esc)
  \t\t\t\t\t"PKMID_T" : 63.75, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1980 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 12.2578125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 9.265625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1861, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 6, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 12.7109375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 10.4375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1974, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 7, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.375, (esc)
  \t\t\t\t\t"PKMID_C" : 157.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 297, (esc)
  \t\t\t\t\t"PKMID_G" : 100.3125, (esc)
  \t\t\t\t\t"PKMID_T" : 63, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 13.390625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 11.375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1861, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 8, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 13.7890625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 12.359375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1974, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 9, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.125, (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 297, (esc)
  \t\t\t\t\t"PKMID_G" : 100.3125, (esc)
  \t\t\t\t\t"PKMID_T" : 63, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 14.46875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 13.1015625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1861, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 10, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 14.8671875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 13.9453125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1974, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 11, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 297, (esc)
  \t\t\t\t\t"PKMID_G" : 100.25, (esc)
  \t\t\t\t\t"PKMID_T" : 62.9375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 15.546875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 14.5546875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1861, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 12, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 15.890625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 15.34375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1974, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 13, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 383, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 297, (esc)
  \t\t\t\t\t"PKMID_G" : 100.25, (esc)
  \t\t\t\t\t"PKMID_T" : 62.875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 1, (esc)
  \t\t\t"ZMW_NUMBER" : 1 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }

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
  \t\t\t\t\t"BASELINE_MEAN" : 2.9921875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 6.18359375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2067, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 19, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 34, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.625, (esc)
  \t\t\t\t\t"PKMID_C" : 158.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 631, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 304, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 396, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 544, (esc)
  \t\t\t\t\t"PKMID_G" : 103.5, (esc)
  \t\t\t\t\t"PKMID_T" : 65.0625, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2071 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"ACTIVITY_LABEL" : 0, (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 10.0703125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 14.1328125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 21880, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 186, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 224, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 280, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 306, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1014, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.25, (esc)
  \t\t\t\t\t"PKMID_C" : 156.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 5581, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3942, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5296, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5069, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 1.5927734375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.74609375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 215, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0.00080156326293945312, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.00087022781372070312, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.00103759765625, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00301361083984375, (esc)
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
  \t\t\t\t\t"TRACE_AUTOCORR" : -0.59912109375 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 3.568359375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.89453125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2023, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.0013294219970703125, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00208282470703125, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 34, (esc)
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
  \t\t\t\t\t"PKMID_FRAMES_T" : 645, (esc)
  \t\t\t\t\t"PKMID_G" : 102.625, (esc)
  \t\t\t\t\t"PKMID_T" : 66.4375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.1328125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2025, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 5.98828125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 6.37890625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2056, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.00056982040405273438, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.0014925003051757812, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00046682357788085938, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 22, (esc)
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
  \t\t\t\t\t"PKMID_FRAMES_T" : 566, (esc)
  \t\t\t\t\t"PKMID_G" : 101.875, (esc)
  \t\t\t\t\t"PKMID_T" : 65.5625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.109375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2056, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 8.8046875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 6.3046875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2012, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.0002307891845703125, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0.0011844635009765625, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00066804885864257812, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 34, (esc)
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
  \t\t\t\t\t"PKMID_FRAMES_T" : 639, (esc)
  \t\t\t\t\t"PKMID_G" : 100.625, (esc)
  \t\t\t\t\t"PKMID_T" : 64.6875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.02734375, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2012, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
