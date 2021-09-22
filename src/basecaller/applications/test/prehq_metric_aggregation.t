This cram test uses experimental configurations that are likely to be changed or removed, once
the real preHQ algorithm is brought on line.  The whole test can probably be removed once
maintenance becomes annoying, as long as something else is dropped in its place

  $ BAZFILE=tmp.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --numZmwLanes 4 --config multipleBazFiles=false --config layout.lanesPerPool=1 --frames=57344 --config=algorithm.modelEstimationMode=FixedEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} --config=prelimHQ.enablePreHQ=true --config=prelimHQ.hqThrottleFraction=1.0 > /dev/null

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
  \t\t\t\t\t"BASE_WIDTH" : 28304, (esc)
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
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6447, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5376, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5305, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 8364, (esc)
  \t\t\t\t\t"PKMID_G" : 98.9375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.0625, (esc)
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

# Run again but set the hqThrottleFraction so start of HQ-region is different. 
  $ smrt-basecaller --numZmwLanes 4 --config multipleBazFiles=false --config layout.lanesPerPool=1 --frames=57344 --config=algorithm.modelEstimationMode=FixedEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} --config=prelimHQ.enablePreHQ=true --config=prelimHQ.hqThrottleFraction=0.25 > /dev/null

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
  \t\t\t\t\t"BASE_WIDTH" : 28304, (esc)
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
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 157, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6447, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 5376, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 5305, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 8364, (esc)
  \t\t\t\t\t"PKMID_G" : 98.9375, (esc)
  \t\t\t\t\t"PKMID_T" : 62.0625, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 0.11749267578125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 5.890625, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : 0.42431640625, (esc)
  \t\t\t\t\t"BASELINE_SD" : 16.765625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 24868, (esc)
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
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : "Inf", (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 6999, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 6404, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 4464, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 4428, (esc)
  \t\t\t\t\t"PKMID_G" : 100.625, (esc)
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
  \t\t\t\t\t"PKMID_C" : 156.5, (esc)
  \t\t\t\t\t"PKMID_FRAMES_A" : 930, (esc)
  \t\t\t\t\t"PKMID_FRAMES_C" : 650, (esc)
  \t\t\t\t\t"PKMID_FRAMES_G" : 880, (esc)
  \t\t\t\t\t"PKMID_FRAMES_T" : 844, (esc)
  \t\t\t\t\t"PKMID_G" : 99.625, (esc)
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
  \t\t\t\t\t"BASELINE_MEAN" : "Inf", (esc)
  \t\t\t\t\t"BASELINE_SD" : 12.3515625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 21894, (esc)
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
  \t\t\t\t\t"PKMID_A" : "Inf", (esc)
  \t\t\t\t\t"PKMID_C" : 156.5, (esc)
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
  \t\t\t\t\t"NUM_PULSES" : 100, (esc)
  \t\t\t\t\t"NUM_PULSE_LABEL_STUTTERS" : 0, (esc)
  \t\t\t\t\t"NUM_SANDWICHES" : 0, (esc)
  \t\t\t\t\t"PIXEL_CHECKSUM" : 0, (esc)
  \t\t\t\t\t"PKMAX_A" : 0, (esc)
  \t\t\t\t\t"PKMAX_C" : 0, (esc)
  \t\t\t\t\t"PKMAX_G" : 0, (esc)
  \t\t\t\t\t"PKMAX_T" : 0, (esc)
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
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1926, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.68798828125, (esc)
  \t\t\t\t\t"BASELINE_SD" : 7.1953125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2108, (esc)
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
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.48193359375, (esc)
  \t\t\t\t\t"BASELINE_SD" : 9.3203125, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 1937, (esc)
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
  \t\t\t\t\t"PULSE_DETECTION_SCORE" : 0, (esc)
  \t\t\t\t\t"PULSE_WIDTH" : 1937, (esc)
  \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
  \t\t\t\t}, (esc)
  \t\t\t\t{ (esc)
  \t\t\t\t\t"BASELINE_MEAN" : -0.9326171875, (esc)
  \t\t\t\t\t"BASELINE_SD" : 10.9140625, (esc)
  \t\t\t\t\t"BASE_WIDTH" : 2108, (esc)
  \t\t\t\t\t"BPZVAR_A" : 0, (esc)
  \t\t\t\t\t"BPZVAR_C" : 0, (esc)
  \t\t\t\t\t"BPZVAR_G" : 0, (esc)
  \t\t\t\t\t"BPZVAR_T" : 0, (esc)
  \t\t\t\t\t"DME_STATUS" : 0, (esc)
  \t\t\t\t\t"MF_ID" : 3, (esc)
  \t\t\t\t\t"NUM_BASES_A" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_C" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_G" : 0, (esc)
  \t\t\t\t\t"NUM_BASES_T" : 0, (esc)
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
