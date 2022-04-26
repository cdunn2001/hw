  $ TRCIN=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.csvOutputFile="rtmetrics.csv" \
  > --config realTimeMetrics.useSingleActivityLabels=false \
  > --config realTimeMetrics.regions='[{"name":"TR","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","Pkmid","SNR","PulseRate","PulseWidth","BaseRate","BaseWidth"]}]' > /dev/null 2>&1

  $ head -n1 rtmetrics.csv
  StartFrame,NumFrames,StartFrameTS,EndFrameTS,TR_Baseline_sampleCV,TR_Baseline_sampleMean,TR_Baseline_sampleMed,TR_Baseline_sampleSize,TR_Baseline_sampleTotal,TR_BaselineStd_sampleCV,TR_BaselineStd_sampleMean,TR_BaselineStd_sampleMed,TR_BaselineStd_sampleSize,TR_BaselineStd_sampleTotal,TR_Pkmid_A_sampleCV,TR_Pkmid_C_sampleCV,TR_Pkmid_G_sampleCV,TR_Pkmid_T_sampleCV,TR_Pkmid_A_sampleMean,TR_Pkmid_C_sampleMean,TR_Pkmid_G_sampleMean,TR_Pkmid_T_sampleMean,TR_Pkmid_A_sampleMed,TR_Pkmid_C_sampleMed,TR_Pkmid_G_sampleMed,TR_Pkmid_T_sampleMed,TR_Pkmid_A_sampleSize,TR_Pkmid_C_sampleSize,TR_Pkmid_G_sampleSize,TR_Pkmid_T_sampleSize,TR_Pkmid_A_sampleTotal,TR_Pkmid_C_sampleTotal,TR_Pkmid_G_sampleTotal,TR_Pkmid_T_sampleTotal,TR_SNR_A_sampleCV,TR_SNR_C_sampleCV,TR_SNR_G_sampleCV,TR_SNR_T_sampleCV,TR_SNR_A_sampleMean,TR_SNR_C_sampleMean,TR_SNR_G_sampleMean,TR_SNR_T_sampleMean,TR_SNR_A_sampleMed,TR_SNR_C_sampleMed,TR_SNR_G_sampleMed,TR_SNR_T_sampleMed,TR_SNR_A_sampleSize,TR_SNR_C_sampleSize,TR_SNR_G_sampleSize,TR_SNR_T_sampleSize,TR_SNR_A_sampleTotal,TR_SNR_C_sampleTotal,TR_SNR_G_sampleTotal,TR_SNR_T_sampleTotal,TR_PulseRate_sampleCV,TR_PulseRate_sampleMean,TR_PulseRate_sampleMed,TR_PulseRate_sampleSize,TR_PulseRate_sampleTotal,TR_PulseWidth_sampleCV,TR_PulseWidth_sampleMean,TR_PulseWidth_sampleMed,TR_PulseWidth_sampleSize,TR_PulseWidth_sampleTotal,TR_BaseRate_sampleCV,TR_BaseRate_sampleMean,TR_BaseRate_sampleMed,TR_BaseRate_sampleSize,TR_BaseRate_sampleTotal,TR_BaseWidth_sampleCV,TR_BaseWidth_sampleMean,TR_BaseWidth_sampleMed,TR_BaseWidth_sampleSize,TR_BaseWidth_sampleTotal

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.csvOutputFile="rtmetrics.csv" \
  > --config realTimeMetrics.useSingleActivityLabels=false \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","SNR"]}]' > /dev/null 2>&1

  $ wc -l rtmetrics.csv
  4 rtmetrics.csv

  $ cat rtmetrics.csv
  StartFrame,NumFrames,StartFrameTS,EndFrameTS,TestRegion_Baseline_sampleCV,TestRegion_Baseline_sampleMean,TestRegion_Baseline_sampleMed,TestRegion_Baseline_sampleSize,TestRegion_Baseline_sampleTotal,TestRegion_BaselineStd_sampleCV,TestRegion_BaselineStd_sampleMean,TestRegion_BaselineStd_sampleMed,TestRegion_BaselineStd_sampleSize,TestRegion_BaselineStd_sampleTotal,TestRegion_SNR_A_sampleCV,TestRegion_SNR_C_sampleCV,TestRegion_SNR_G_sampleCV,TestRegion_SNR_T_sampleCV,TestRegion_SNR_A_sampleMean,TestRegion_SNR_C_sampleMean,TestRegion_SNR_G_sampleMean,TestRegion_SNR_T_sampleMean,TestRegion_SNR_A_sampleMed,TestRegion_SNR_C_sampleMed,TestRegion_SNR_G_sampleMed,TestRegion_SNR_T_sampleMed,TestRegion_SNR_A_sampleSize,TestRegion_SNR_C_sampleSize,TestRegion_SNR_G_sampleSize,TestRegion_SNR_T_sampleSize,TestRegion_SNR_A_sampleTotal,TestRegion_SNR_C_sampleTotal,TestRegion_SNR_G_sampleTotal,TestRegion_SNR_T_sampleTotal
  20456,4096,0,0,117.06*,0.79*,-1,4096,4096,64.0*,5.84*,-1,4096,4096,64.0*,64.0*,64.0*,64.0*,40.39*,27.28*,17.57*,11.11*,-1,-1,-1,-1,4096,4096,4096,4096,4096,4096,4096,4096 (glob)
  24552,4096,0,0,121.72*,0.74*,-1,4096,4096,64.0*,5.85*,-1,4096,4096,64.0*,64.0*,64.0*,64.0*,40.39*,27.28*,17.56*,11.10*,-1,-1,-1,-1,4096,4096,4096,4096,4096,4096,4096,4096 (glob)
  28648,4096,0,0,121.86*,0.73*,-1,4096,4096,64.0*,5.85*,-1,4096,4096,64.0*,64.0*,64.0*,64.0*,40.39*,27.28*,17.56*,11.11*,-1,-1,-1,-1,4096,4096,4096,4096,4096,4096,4096,4096 (glob)

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.jsonOutputFile="rtmetrics.json" \
  > --config realTimeMetrics.useSingleActivityLabels=false \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","SNR"]}]' > /dev/null 2>&1

  $ wc -l rtmetrics.json
  1 rtmetrics.json

  $ cat rtmetrics.json | python -m json.tool
  {
      "frameTimeStampDelta": 0,
      "metricsChunk": {
          "metricsBlocks": [
              {
                  "beginFrameTimeStamp": 0,
                  "endFrameTimeStamp": 0,
                  "groups": [
                      {
                          "metrics": [
                              {
                                  "name": "Baseline",
                                  "sampleCV": [
                                      121.86* (glob)
                                  ],
                                  "sampleMean": [
                                      0.73* (glob)
                                  ],
                                  "sampleMed": [
                                      -1
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
                                      64.0* (glob)
                                  ],
                                  "sampleMean": [
                                      5.85* (glob)
                                  ],
                                  "sampleMed": [
                                      -1
                                  ],
                                  "sampleSize": [
                                      4096
                                  ],
                                  "sampleTotal": [
                                      4096
                                  ]
                              },
                              {
                                  "name": "SNR",
                                  "sampleCV": [
                                      64.04*, (glob)
                                      64.04*, (glob)
                                      64.05*, (glob)
                                      64.06* (glob)
                                  ],
                                  "sampleMean": [
                                      40.39*, (glob)
                                      27.28*, (glob)
                                      17.56*, (glob)
                                      11.11* (glob)
                                  ],
                                  "sampleMed": [
                                      -1,
                                      -1,
                                      -1,
                                      -1
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
                  "startFrame": 28648
              }
          ],
          "numMetricsBlocks": 1
      },
      "startFrameTimeStamp": 0,
      "token": ""
  }
