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
  20456,4096,0,0,63.992240905761719,299.85250854492188,-1,4096,4096,64.035758972167969,5.8483366966247559,-1,4096,4096,64.045433044433594,64.04840087890625,64.054000854492188,64.063003540039062,40.396106719970703,27.280050277709961,17.573976516723633,11.117399215698242,-1,-1,-1,-1,4096,4096,4096,4096,4096,4096,4096,4096
  24552,4096,0,0,63.992237091064453,299.85269165039062,-1,4096,4096,64.035942077636719,5.8501648902893066,-1,4096,4096,64.045928955078125,64.048858642578125,64.05487060546875,64.063514709472656,40.390090942382812,27.281242370605469,17.568403244018555,11.109755516052246,-1,-1,-1,-1,4096,4096,4096,4096,4096,4096,4096,4096
  28648,4096,0,0,63.992240905761719,299.85159301757812,-1,4096,4096,64.035903930664062,5.850440502166748,-1,4096,4096,64.045928955078125,64.048912048339844,64.054794311523438,64.063522338867188,40.390132904052734,27.284442901611328,17.569520950317383,11.110082626342773,-1,-1,-1,-1,4096,4096,4096,4096,4096,4096,4096,4096

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
                                      63.99224090576172
                                  ],
                                  "sampleMean": [
                                      299.8515930175781
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
