  $ TRCIN=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.rtMetricsFile="rtmetrics.json" \
  > --config realTimeMetrics.useSingleActivityLabels=false \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,0,64,64]]}]' > /dev/null 2>&1

  $ wc -l rtmetrics.json
  3 rtmetrics.json

  $ head -n1 rtmetrics.json | python -m json.tool | grep -A6 baseRate
          "baseRate": {
              "sampleCV": 63.9*, (glob)
              "sampleMean": 0.06*, (glob)
              "sampleMedian": null,
              "sampleSize": 4096,
              "sampleTotal": 4096
          },

  $ head -n1 rtmetrics.json | python -m json.tool | grep -A6 baseline
          "baseline": {
              "sampleCV": 116.8*, (glob)
              "sampleMean": 0.79*, (glob)
              "sampleMedian": null,
              "sampleSize": 4096,
              "sampleTotal": 4096
          },
          "baselineSd": {
              "sampleCV": 64.0*, (glob)
              "sampleMean": 5.8*, (glob)
              "sampleMedian": null,
              "sampleSize": 4096,
              "sampleTotal": 4096
          },

  $ head -n+1 rtmetrics.json | python -m json.tool | grep -A29 snr
          "snr": [
              {
                  "sampleCV": 64.0*, (glob)
                  "sampleMean": 40.3*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              },
              {
                  "sampleCV": 64.0*, (glob)
                  "sampleMean": 27.2*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              },
              {
                  "sampleCV": 64.0*, (glob)
                  "sampleMean": 17.5*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              },
              {
                  "sampleCV": 64.0*, (glob)
                  "sampleMean": 11.1*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              }
          ],
