  $ TRCIN=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.csvOutputFile="rtmetrics.csv" \
  > --config realTimeMetrics.regions='[{"name":"TR","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","Pkmid","Snr","PulseRate","PulseWidth","BaseRate","BaseWidth"],"featuresForFilter":["Sequencing"]}]' > /dev/null 2>&1

  $ head -n1 rtmetrics.csv
  StartFrame,NumFrames,StartFrameTS,EndFrameTS,TR_Baseline_sampleCV,TR_Baseline_sampleMean,TR_Baseline_sampleMed,TR_Baseline_sampleSize,TR_Baseline_sampleTotal,TR_BaselineStd_sampleCV,TR_BaselineStd_sampleMean,TR_BaselineStd_sampleMed,TR_BaselineStd_sampleSize,TR_BaselineStd_sampleTotal,TR_Pkmid_A_sampleCV,TR_Pkmid_C_sampleCV,TR_Pkmid_G_sampleCV,TR_Pkmid_T_sampleCV,TR_Pkmid_A_sampleMean,TR_Pkmid_C_sampleMean,TR_Pkmid_G_sampleMean,TR_Pkmid_T_sampleMean,TR_Pkmid_A_sampleMed,TR_Pkmid_C_sampleMed,TR_Pkmid_G_sampleMed,TR_Pkmid_T_sampleMed,TR_Pkmid_A_sampleSize,TR_Pkmid_C_sampleSize,TR_Pkmid_G_sampleSize,TR_Pkmid_T_sampleSize,TR_Pkmid_A_sampleTotal,TR_Pkmid_C_sampleTotal,TR_Pkmid_G_sampleTotal,TR_Pkmid_T_sampleTotal,TR_Snr_A_sampleCV,TR_Snr_C_sampleCV,TR_Snr_G_sampleCV,TR_Snr_T_sampleCV,TR_Snr_A_sampleMean,TR_Snr_C_sampleMean,TR_Snr_G_sampleMean,TR_Snr_T_sampleMean,TR_Snr_A_sampleMed,TR_Snr_C_sampleMed,TR_Snr_G_sampleMed,TR_Snr_T_sampleMed,TR_Snr_A_sampleSize,TR_Snr_C_sampleSize,TR_Snr_G_sampleSize,TR_Snr_T_sampleSize,TR_Snr_A_sampleTotal,TR_Snr_C_sampleTotal,TR_Snr_G_sampleTotal,TR_Snr_T_sampleTotal,TR_PulseRate_sampleCV,TR_PulseRate_sampleMean,TR_PulseRate_sampleMed,TR_PulseRate_sampleSize,TR_PulseRate_sampleTotal,TR_PulseWidth_sampleCV,TR_PulseWidth_sampleMean,TR_PulseWidth_sampleMed,TR_PulseWidth_sampleSize,TR_PulseWidth_sampleTotal,TR_BaseRate_sampleCV,TR_BaseRate_sampleMean,TR_BaseRate_sampleMed,TR_BaseRate_sampleSize,TR_BaseRate_sampleTotal,TR_BaseWidth_sampleCV,TR_BaseWidth_sampleMean,TR_BaseWidth_sampleMed,TR_BaseWidth_sampleSize,TR_BaseWidth_sampleTotal

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.csvOutputFile="rtmetrics.csv" \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","Snr"],"featuresForFilter":["Sequencing"],"useSingleActivityLabels":false}]' > /dev/null 2>&1

  $ wc -l rtmetrics.csv
  4 rtmetrics.csv

  $ cat rtmetrics.csv
  StartFrame,NumFrames,StartFrameTS,EndFrameTS,TestRegion_Baseline_sampleCV,TestRegion_Baseline_sampleMean,TestRegion_Baseline_sampleMed,TestRegion_Baseline_sampleSize,TestRegion_Baseline_sampleTotal,TestRegion_BaselineStd_sampleCV,TestRegion_BaselineStd_sampleMean,TestRegion_BaselineStd_sampleMed,TestRegion_BaselineStd_sampleSize,TestRegion_BaselineStd_sampleTotal,TestRegion_Snr_A_sampleCV,TestRegion_Snr_C_sampleCV,TestRegion_Snr_G_sampleCV,TestRegion_Snr_T_sampleCV,TestRegion_Snr_A_sampleMean,TestRegion_Snr_C_sampleMean,TestRegion_Snr_G_sampleMean,TestRegion_Snr_T_sampleMean,TestRegion_Snr_A_sampleMed,TestRegion_Snr_C_sampleMed,TestRegion_Snr_G_sampleMed,TestRegion_Snr_T_sampleMed,TestRegion_Snr_A_sampleSize,TestRegion_Snr_C_sampleSize,TestRegion_Snr_G_sampleSize,TestRegion_Snr_T_sampleSize,TestRegion_Snr_A_sampleTotal,TestRegion_Snr_C_sampleTotal,TestRegion_Snr_G_sampleTotal,TestRegion_Snr_T_sampleTotal
  20456,4096,0,40960000,63\.99\d*,299\.85\d*,299\.8\d*,4096,4096,64\.03\d*,5\.8[45]\d*,5\.84\d*,4096,4096,64\.04\d*,64\.04\d*,64\.05\d*,64\.06\d*,40\.39\d*,27\.2[89]\d*,17\.58\d*,11\.11\d*,40\.3[23]\d*,27\.26\d*,17\.56\d*,11\.10\d*,4096,4096,4096,4096,4096,4096,4096,4096 (re)
  24552,4096,0,40960000,63\.99\d*,299\.85\d*,299\.8\d*,4096,4096,64\.03\d*,5\.8[45]\d*,5\.84\d*,4096,4096,64\.04\d*,64\.04\d*,64\.05\d*,64\.06\d*,40\.39\d*,27\.2[89]\d*,17\.57\d*,11\.11\d*,40\.32\d*,27\.26\d*,17\.56\d*,11\.10\d*,4096,4096,4096,4096,4096,4096,4096,4096 (re)
  28648,4096,0,40960000,63\.99\d*,299\.85\d*,299\.8\d*,4096,4096,64\.03\d*,5\.8[45]\d*,5\.8[45]\d*,4096,4096,64\.04\d*,64\.04\d*,64\.05\d*,64\.06\d*,40\.39\d*,27\.2[89]\d*,17\.57\d*,11\.11\d*,40\.32\d*,27\.2[45]\d*,17\.55\d*,11\.10\d*,4096,4096,4096,4096,4096,4096,4096,4096 (re)

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.jsonOutputFile="rtmetrics.json" \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","Snr"],"featuresForFilter":["Sequencing"],"useSingleActivityLabels":false}]' > /dev/null 2>&1

  $ wc -l rtmetrics.json
  1 rtmetrics.json

  $ cat rtmetrics.json | python -m json.tool
  {
      "frameTimeStampDelta": 10000,
      "metricsChunk": {
          "metricsBlocks": [
              {
                  "endFrameTimeStamp": 40960000,
                  "groups": [
                      {
                          "metrics": [
                              {
                                  "name": "Baseline",
                                  "sampleCV": [
                                      63.99* (glob)
                                  ],
                                  "sampleMean": [
                                      299.85* (glob)
                                  ],
                                  "sampleMed": [
                                      299.84* (glob)
                                  ],
                                  "sampleSize": [
                                      4096
                                  ],
                                  "sampleTotal": [
                                      4096
                                  ]
                              },
                              {
                                  "name": "BaselineStd",
                                  "sampleCV": [
                                      64.03* (glob)
                                  ],
                                  "sampleMean": [
                                      5\.8[45]\d* (re)
                                  ],
                                  "sampleMed": [
                                      5\.8[45]\d* (re)
                                  ],
                                  "sampleSize": [
                                      4096
                                  ],
                                  "sampleTotal": [
                                      4096
                                  ]
                              },
                              {
                                  "name": "Snr",
                                  "sampleCV": [
                                      64.04* (glob)
                                      64.04* (glob)
                                      64.05* (glob)
                                      64.06* (glob)
                                  ],
                                  "sampleMean": [
                                      40.39* (glob)
                                      27\.2[89]\d*, (re)
                                      17.57* (glob)
                                      11.11* (glob)
                                  ],
                                  "sampleMed": [
                                      40.32* (glob)
                                      27\.2[456]\d*, (re)
                                      17.55* (glob)
                                      11.10* (glob)
                                  ],
                                  "sampleSize": [
                                      4096,
                                      4096,
                                      4096,
                                      4096
                                  ],
                                  "sampleTotal": [
                                      4096,
                                      4096,
                                      4096,
                                      4096
                                  ]
                              }
                          ],
                          "region": "TestRegion"
                      }
                  ],
                  "numFrames": 4096,
                  "startFrame": 28648,
                  "startFrameTimeStamp": 0
              }
          ],
          "numMetricsBlocks": 1
      },
      "startFrameTimeStamp": 0,
      "token": ""
  }
