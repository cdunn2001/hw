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
  \t\t\t\t\t"BASELINE_MEAN" : 0, (esc)
  \t\t\t\t\t"BASELINE_SD" : 0, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 0, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 0, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 0, (esc)
  \t\t\t\t\t"PKMID_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 0, (esc)
  \t\t\t\t\t"PKMID_G" : 0, (esc)
  \t\t\t\t\t"PKMID_T" : 0, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 0, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 6.29296875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 13.8046875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 26537, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 319, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 314, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 371, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 53760, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1320, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 228.375, (esc)
  \t\t\t\t\t"PKMID_C" : 157.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6058, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5059, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4927, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 7855, (esc)
  \t\t\t\t\t"PKMID_G" : 99.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 26539, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 0, (esc)
  \t\t\t\t\t"BASELINE_SD" : 0, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 0, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 0, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 0, (esc)
  \t\t\t\t\t"PKMID_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 0, (esc)
  \t\t\t\t\t"PKMID_G" : 0, (esc)
  \t\t\t\t\t"PKMID_T" : 0, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 0, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 6.29296875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 13.8046875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 26537, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 315, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 319, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 314, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 371, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 53760, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1320, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 228.375, (esc)
  \t\t\t\t\t"PKMID_C" : 157.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6058, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5059, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4927, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 7855, (esc)
  \t\t\t\t\t"PKMID_G" : 99.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 26539, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 0.11553955078125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.14453125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 217, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 6, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 3, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 512, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 12, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 226.5, (esc)
  \t\t\t\t\t"PKMID_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 11, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 0, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 112, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 70, (esc)
  \t\t\t\t\t"PKMID_G" : 100, (esc)
  \t\t\t\t\t"PKMID_T" : 64.25, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 217, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 2.33203125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.98828125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1982, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 20, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.875, (esc)
  \t\t\t\t\t"PKMID_C" : 158.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 556, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 545, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 384, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 301, (esc)
  \t\t\t\t\t"PKMID_G" : 102.5, (esc)
  \t\t\t\t\t"PKMID_T" : 65.5, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1982, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 3.9375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.44140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1867, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.75, (esc)
  \t\t\t\t\t"PKMID_C" : 159.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 526, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 427, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 328, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 387, (esc)
  \t\t\t\t\t"PKMID_G" : 103.0625, (esc)
  \t\t\t\t\t"PKMID_T" : 64.4375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1867, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 5.703125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 8.765625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1974, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.75, (esc)
  \t\t\t\t\t"PKMID_C" : 157.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 555, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 543, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 382, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 300, (esc)
  \t\t\t\t\t"PKMID_G" : 101.25, (esc)
  \t\t\t\t\t"PKMID_T" : 64.125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1976, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 6.83984375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 10.140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1865, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.75, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 526, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 427, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 327, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 386, (esc)
  \t\t\t\t\t"PKMID_G" : 102, (esc)
  \t\t\t\t\t"PKMID_T" : 63.5, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1865, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 7.703125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 11.2421875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1972, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 5, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.875, (esc)
  \t\t\t\t\t"PKMID_C" : 156.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 543, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 382, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 299, (esc)
  \t\t\t\t\t"PKMID_G" : 100.3125, (esc)
  \t\t\t\t\t"PKMID_T" : 63.21875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1974, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 8.2578125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 12.53125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1863, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 6, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.25, (esc)
  \t\t\t\t\t"PKMID_C" : 158, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 426, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 327, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 386, (esc)
  \t\t\t\t\t"PKMID_G" : 101.5625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.15625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1863, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 8.7109375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 13.609375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1971, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 7, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.625, (esc)
  \t\t\t\t\t"PKMID_C" : 156.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 543, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 382, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 298, (esc)
  \t\t\t\t\t"PKMID_G" : 100.25, (esc)
  \t\t\t\t\t"PKMID_T" : 63.09375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1973, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 8.953125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 14.8203125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1862, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 8, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 157.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 426, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 327, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 385, (esc)
  \t\t\t\t\t"PKMID_G" : 101.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1862, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 9.1015625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 15.8828125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1971, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 9, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.375, (esc)
  \t\t\t\t\t"PKMID_C" : 156.375, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 543, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 382, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 298, (esc)
  \t\t\t\t\t"PKMID_G" : 100, (esc)
  \t\t\t\t\t"PKMID_T" : 62.78125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1973, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 9.328125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 16.9375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1862, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 10, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.875, (esc)
  \t\t\t\t\t"PKMID_C" : 157.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 426, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 327, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 385, (esc)
  \t\t\t\t\t"PKMID_G" : 101.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1862, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 9.4765625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 17.875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1971, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 11, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.375, (esc)
  \t\t\t\t\t"PKMID_C" : 156.375, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 543, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 382, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 298, (esc)
  \t\t\t\t\t"PKMID_G" : 100, (esc)
  \t\t\t\t\t"PKMID_T" : 62.78125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1973, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 9.703125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 18.8125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1862, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 12, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 30, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 32, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 15, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.875, (esc)
  \t\t\t\t\t"PKMID_C" : 157.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 525, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 426, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 327, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 385, (esc)
  \t\t\t\t\t"PKMID_G" : 101.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1862, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 9.8515625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 19.65625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1971, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 13, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 19, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.375, (esc)
  \t\t\t\t\t"PKMID_C" : 156.375, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 543, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 382, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 298, (esc)
  \t\t\t\t\t"PKMID_G" : 100, (esc)
  \t\t\t\t\t"PKMID_T" : 62.78125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1973, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 2.259765625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 6.16796875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2066, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 19, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 34, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4608, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 98, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.125, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 631, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 304, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 396, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 543, (esc)
  \t\t\t\t\t"PKMID_G" : 102.4375, (esc)
  \t\t\t\t\t"PKMID_T" : 64.3125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2070, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 9.5859375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 14.1953125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 21892, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 186, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 224, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 280, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 307, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 49152, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1014, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230.75, (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 5583, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3940, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5284, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5091, (esc)
  \t\t\t\t\t"PKMID_G" : 100.0625, (esc)
  \t\t\t\t\t"PKMID_T" : 63.0625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 21927, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 0.69482421875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.7265625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 215, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 4, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 2, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 512, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 13, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 233.75, (esc)
  \t\t\t\t\t"PKMID_C" : 156.75, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 60, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 36, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 73, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 20, (esc)
  \t\t\t\t\t"PKMID_G" : 95.9375, (esc)
  \t\t\t\t\t"PKMID_T" : 59.78125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 215, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 1.8759765625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.7890625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2014, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 34, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 101, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 228.875, (esc)
  \t\t\t\t\t"PKMID_C" : 159.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 472, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 407, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 292, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 641, (esc)
  \t\t\t\t\t"PKMID_G" : 101.375, (esc)
  \t\t\t\t\t"PKMID_T" : 64.6875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2014, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 2.93359375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.1953125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2050, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 22, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 231.125, (esc)
  \t\t\t\t\t"PKMID_C" : 158.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 449, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 363, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 474, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 564, (esc)
  \t\t\t\t\t"PKMID_G" : 100.25, (esc)
  \t\t\t\t\t"PKMID_T" : 63.75, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2050, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 4.4296875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 8.734375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2008, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 23, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 21, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 34, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 101, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 227.625, (esc)
  \t\t\t\t\t"PKMID_C" : 158.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 472, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 407, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 291, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 636, (esc)
  \t\t\t\t\t"PKMID_G" : 99.8125, (esc)
  \t\t\t\t\t"PKMID_T" : 63.3125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2008, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
