  $ TRCIN=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":64,"traceFile":"'$TRCIN'"}' \
  > --config system.basecallerConcurrency=1 \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 \
  > --config realTimeMetrics.rtMetricsFile="rtmetrics.json" \
  > --config realTimeMetrics.useSingleActivityLabels=false \
  > --config realTimeMetrics.regions='[{"name":"TestRegion","roi":[[0,4096]]}]' > /dev/null 2>&1

  $ wc -l rtmetrics.json
  3 rtmetrics.json

  $ head -n1 rtmetrics.json | python -m json.tool | grep -A6 baseRate
          "baseRate": {
              "sampleCV": 63.99*, (glob)
              "sampleMean": 0.0625*, (glob)
              "sampleMedian": null,
              "sampleSize": 4096,
              "sampleTotal": 4096
          },

  $ head -n1 rtmetrics.json | python -m json.tool | grep -A6 baseline
          "baseline": {
              "sampleCV": 116.85*, (glob)
              "sampleMean": 0.794*, (glob)
              "sampleMedian": null,
              "sampleSize": 4096,
              "sampleTotal": 4096
          },
          "baselineSd": {
              "sampleCV": 64.03*, (glob)
              "sampleMean": 5.84*, (glob)
              "sampleMedian": null,
              "sampleSize": 4096,
              "sampleTotal": 4096
          },

  $ head -n+1 rtmetrics.json | python -m json.tool | grep -A29 snr
          "snr": [
              {
                  "sampleCV": 64.04*, (glob)
                  "sampleMean": 40.39*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              },
              {
                  "sampleCV": 64.04*, (glob)
                  "sampleMean": 27.28*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              },
              {
                  "sampleCV": 64.05*, (glob)
                  "sampleMean": 17.58*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              },
              {
                  "sampleCV": 64.06*, (glob)
                  "sampleMean": 11.12*, (glob)
                  "sampleMedian": null,
                  "sampleSize": 4096,
                  "sampleTotal": 4096
              }
          ],
