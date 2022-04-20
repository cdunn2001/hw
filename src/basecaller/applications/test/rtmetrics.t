  $ TRCIN=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.rtMetricsFile="rtmetrics.json" \
  > --config realTimeMetrics.useSingleActivityLabels=false \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,0,64,64]],"metrics":["Baseline","BaselineStd","Pkmid","SNR","PulseRate","PulseWidth","BaseRate","BaseWidth"]}]' > /dev/null 2>&1


  $ wc -l rtmetrics.json
  3 rtmetrics.json

  $ head -n1 rtmetrics.json | python -m json.tool | grep -B1 -A16 BaseRate
                              {
                                  "name": "BaseRate",
                                  "sampleCV": [
                                      63.9* (glob)
                                  ],
                                  "sampleMean": [
                                      0.06* (glob)
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
  $ head -n1 rtmetrics.json | python -m json.tool | grep -B1 -A16 Baseline
                              {
                                  "name": "Baseline",
                                  "sampleCV": [
                                      117.0* (glob)
                                  ],
                                  "sampleMean": [
                                      0.79* (glob)
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
                                      5.8* (glob)
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

  $ head -n+1 rtmetrics.json | python -m json.tool | grep -B1 -A31 SNR
                              {
                                  "name": "SNR",
                                  "sampleCV": [
                                      64.0*, (glob)
                                      64.0*, (glob)
                                      64.0*, (glob)
                                      64.0* (glob)
                                  ],
                                  "sampleMean": [
                                      40.3*, (glob)
                                      27.2*, (glob)
                                      17.5*, (glob)
                                      11.1* (glob)
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
                              },
