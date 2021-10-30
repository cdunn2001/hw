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
  \t\t\t"MF_METRICS" :  (esc)
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
  \t\t\t\t\t"MF_ID" : 0, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : -1.2236328125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 15.6484375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 28290, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 336, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 343, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 335, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 385, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 57344, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1406, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 227.75, (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6447, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5376, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5305, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 8364, (esc)
  \t\t\t\t\t"PKMID_G" : 98.8125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.09375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 28304, (esc)
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
  \t\t\t"MF_METRICS" :  (esc)
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
  \t\t\t\t\t"MF_ID" : 0, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : -1.2236328125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 15.6484375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 28290, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 336, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 343, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 335, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 385, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 57344, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1406, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 227.75, (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6447, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5376, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5305, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 8364, (esc)
  \t\t\t\t\t"PKMID_G" : 98.8125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.09375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 28304, (esc)
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
  \t\t\t"MF_METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.1175537109375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.890625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1982, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 27, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 21, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 102, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.25, (esc)
  \t\t\t\t\t"PKMID_C" : 155, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 374, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 492, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 360, (esc)
  \t\t\t\t\t"PKMID_G" : 99.9375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.9375, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1984, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.424072265625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 16.765625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 24842, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 372, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 394, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 294, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 217, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 53248, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1290, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 229.625, (esc)
  \t\t\t\t\t"PKMID_C" : 156.75, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6999, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 6404, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4464, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 4428, (esc)
  \t\t\t\t\t"PKMID_G" : 100.5, (esc)
  \t\t\t\t\t"PKMID_T" : 62.46875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 24868, (esc)
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
  \t\t\t"MF_METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.300048828125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.21875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 3634, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 31, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 37, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 46, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 51, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 8192, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 168, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : 230, (esc)
  \t\t\t\t\t"PKMID_C" : 156.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 930, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 650, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 880, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 844, (esc)
  \t\t\t\t\t"PKMID_G" : 99.5625, (esc)
  \t\t\t\t\t"PKMID_T" : 62.71875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 3640, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 3.38671875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 17.3125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 21858, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 186, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 228, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 276, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 306, (esc)
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
  \t\t\t\t\t"PKMID_A" : 230.25, (esc)
  \t\t\t\t\t"PKMID_C" : 156.625, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 5580, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3936, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5280, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5070, (esc)
  \t\t\t\t\t"PKMID_G" : 99.625, (esc)
  \t\t\t\t\t"PKMID_T" : 62.71875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 21894, (esc)
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
  \t\t\t"MF_METRICS" :  (esc)
  \t\t\t[ (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.4140625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.80859375, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1924, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 26, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 19, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 30, (esc)
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
  \t\t\t\t\t"PKMID_A" : 226.75, (esc)
  \t\t\t\t\t"PKMID_C" : 156.875, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 531, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 417, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 278, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 500, (esc)
  \t\t\t\t\t"PKMID_G" : 97.375, (esc)
  \t\t\t\t\t"PKMID_T" : 61.90625, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1926, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.6884765625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.1953125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 22, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 25, (esc)
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
  \t\t\t\t\t"PKMID_A" : 228.75, (esc)
  \t\t\t\t\t"PKMID_C" : 157.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 390, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 351, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 473, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 694, (esc)
  \t\t\t\t\t"PKMID_G" : 99.5625, (esc)
  \t\t\t\t\t"PKMID_T" : 62.125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.48193359375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 9.3203125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1935, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 26, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 24, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 20, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 30, (esc)
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
  \t\t\t\t\t"PKMID_A" : 227, (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 531, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 417, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 286, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 501, (esc)
  \t\t\t\t\t"PKMID_G" : 97.6875, (esc)
  \t\t\t\t\t"PKMID_T" : 62.03125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1937, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.93310546875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 10.9140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 22, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 25, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 28, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 25, (esc)
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
  \t\t\t\t\t"PKMID_A" : 228.75, (esc)
  \t\t\t\t\t"PKMID_C" : 157.25, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 390, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 351, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 473, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 694, (esc)
  \t\t\t\t\t"PKMID_G" : 99.5625, (esc)
  \t\t\t\t\t"PKMID_T" : 62.125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
