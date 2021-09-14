This cram test uses experimental configurations that are likely to be changed or removed, once
the real preHQ algorithm is brought on line.  The whole test can probably be removed once
maintenance becomes annoying, as long as something else is dropped in its place

  $ BAZFILE=tmp.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --numZmwLanes 4 --config multipleBazFiles=false --config layout.lanesPerPool=1 --frames=81920 --config=algorithm.modelEstimationMode=FixedEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} --config=prelimHQ.enableLookback=true --config=prelimHQ.hqThrottleFraction=1.0 > /dev/null

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
  \t\t\t\t\t"BASELINE_MEAN" : "-Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 20214, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 40960, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1004, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 4605, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3840, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3787, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5974, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 20214, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : "-Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 20225, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 40960, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1005, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 4605, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3840, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3795, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5975, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 20225, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 0, (esc)
  \t\t\t"ZMW_NUMBER" : 0 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }

# Run again but set the hqThrottleFraction so start of HQ-region is different. 
  $ smrt-basecaller --numZmwLanes 4 --config multipleBazFiles=false --config layout.lanesPerPool=1 --frames=81920 --config=algorithm.modelEstimationMode=FixedEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} --config=prelimHQ.enableLookback=true --config=prelimHQ.hqThrottleFraction=0.25 > /dev/null

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
  \t\t\t\t\t"BASELINE_MEAN" : "-Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 20214, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 40960, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1004, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 4605, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3840, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3787, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5974, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 20214, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : "-Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 20225, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 40960, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 1005, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 4605, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3840, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3795, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 5975, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 20225, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 0.11749267578125, (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1984, (esc)
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
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 155, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 554, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 374, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 492, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 360, (esc)
  \t\t\t\t\t"PKMID_G" : 100.125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1984, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : 0.32763671875, (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 19185, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 40960, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 995, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 5395, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 4845, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3540, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 3420, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 19185, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : "Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 17194, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 36864, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 892, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 4841, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 4466, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3048, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 3060, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 17194, (esc)
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
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 3640, (esc)
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
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 930, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 650, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 880, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 844, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : 62.75, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 3640, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : "Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : "Inf", (esc)
  \t\t\t\t\t"BASE_WIDTH" : 18245, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 40960, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 845, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 4650, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 3280, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4400, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 4225, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 18245, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : "Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : 0, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 14596, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 32768, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"NUM_PULSES" : 676, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 3720, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 2624, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 3520, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 3380, (esc)
  \t\t\t\t\t"PKMID_G" : "Inf", (esc)
  \t\t\t\t\t"PKMID_T" : "Inf", (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 14596, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t} (esc)
  \t\t\t], (esc)
  \t\t\t"ZMW_ID" : 2, (esc)
  \t\t\t"ZMW_NUMBER" : 2 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }



# Generate internal metrics by disabling lookback
  $ smrt-basecaller --numZmwLanes 4 --config multipleBazFiles=false --config layout.lanesPerPool=1 --frames=16384 --config=algorithm.modelEstimationMode=FixedEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} > /dev/null

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
  \t\t\t\t\t"BASELINE_SD" : 5.8046875, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1926, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.0012483596801757812, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.003528594970703125, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 3, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 24, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 318, (esc)
  \t\t\t\t\t"PKMAX_C" : 226, (esc)
  \t\t\t\t\t"PKMAX_G" : 137, (esc)
  \t\t\t\t\t"PKMAX_T" : 102, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 156.75, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 531, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 417, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 278, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 500, (esc)
  \t\t\t\t\t"PKMID_G" : 97.5, (esc)
  \t\t\t\t\t"PKMID_T" : 61.875, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.01171875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1926, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.68798828125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.1953125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.003993988037109375, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00094699859619140625, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 1, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 8, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 23, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 313, (esc)
  \t\t\t\t\t"PKMAX_C" : 207, (esc)
  \t\t\t\t\t"PKMAX_G" : 148, (esc)
  \t\t\t\t\t"PKMAX_T" : 97, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 390, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 351, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 473, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 694, (esc)
  \t\t\t\t\t"PKMID_G" : 99.625, (esc)
  \t\t\t\t\t"PKMID_T" : 62.125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.01171875, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.48193359375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 9.3203125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1937, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.000812530517578125, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.0017547607421875, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 2, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 3, (esc)
  \t\t\t\t\t"NUM_PULSES" : 101, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 24, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 318, (esc)
  \t\t\t\t\t"PKMAX_C" : 226, (esc)
  \t\t\t\t\t"PKMAX_G" : 137, (esc)
  \t\t\t\t\t"PKMAX_T" : 102, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 531, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 417, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 286, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 501, (esc)
  \t\t\t\t\t"PKMID_G" : 97.8125, (esc)
  \t\t\t\t\t"PKMID_T" : 62.03125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.01953125, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1937, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.9326171875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 10.9140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0.003810882568359375, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0.00031876564025878906, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
  \t\t\t\t\t"NUM_FRAMES" : 4096, (esc)
  \t\t\t\t\t"NUM_HALF_SANDWICHES" : 8, (esc)
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 23, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 313, (esc)
  \t\t\t\t\t"PKMAX_C" : 207, (esc)
  \t\t\t\t\t"PKMAX_G" : 148, (esc)
  \t\t\t\t\t"PKMAX_T" : 97, (esc)
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 157.125, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 390, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 351, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 473, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 694, (esc)
  \t\t\t\t\t"PKMID_G" : 99.625, (esc)
  \t\t\t\t\t"PKMID_T" : 62.125, (esc)
  \t\t\t\t\t"PKZVAR_A" : 0, (esc)
  \t\t\t\t\t"PKZVAR_C" : 0, (esc)
  \t\t\t\t\t"PKZVAR_G" : 0, (esc)
  \t\t\t\t\t"PKZVAR_T" : 0, (esc)
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : -4.01171875, (esc)
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
