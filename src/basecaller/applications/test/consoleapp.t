  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":1024, "numZmwLanes":1,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config algorithm.Metrics.framesPerHFMetricBlock=512 --config layout.lanesPerPool=1   \
  > --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses \
  > --outputbazfile ${CRAMTMP}/test.baz > /dev/null

  $ bazviewer --silent -l ${CRAMTMP}/test.baz | tail -n +1 | wc -l
  65

  $ bazviewer --silent -d -n 0 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC
  $ bazviewer --silent -d -n 63 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":1024, "numZmwLanes":1,"traceFile":"'$TRCFILE'" }'    \
  > --config multipleBazFiles=false --config algorithm.Metrics.framesPerHFMetricBlock=512 --config=prelimHQ.zmwOutputStride=4 \
  > --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations                                    \
  > --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --outputbazfile ${CRAMTMP}/test.baz > /dev/null

  $ bazviewer --silent -l ${CRAMTMP}/test.baz | tail -n +1 | wc -l
  65

  $ bazviewer --silent -d -n 4 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTAC

  $ bazviewer --silent -d -n 2 ${CRAMTMP}/test.baz  | tail -n +1
  {
  \t"STITCHED" :  (esc)
  \t[ (esc)
  \t\t{ (esc)
  \t\t\t"INTERNAL" : false, (esc)
  \t\t\t"ZMW_ID" : 2, (esc)
  \t\t\t"ZMW_NUMBER" : 2 (esc)
  \t\t} (esc)
  \t], (esc)
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }

  $ head -n15 ${CRAMTMP}/test.baz | grep EXPERIMENT_METADATA
  \t\t"EXPERIMENT_METADATA" : "{\\n   \\"acqParams\\" : {\\n      \\"aduGain\\" : 1,\\n      \\"frameRate\\" : 100,\\n      \\"numFrames\\" : 1024\\n   },\\n   \\"acquisitionXML\\" : \\"ADD_ME\\",\\n   \\"chipInfo\\" : {\\n      \\"analogRefSnr\\" : 40,\\n      \\"imagePsf\\" : [\\n         [ 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 1, 0, 0 ],\\n         [ 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0 ]\\n      ],\\n      \\"layoutName\\" : \\"KestrelPOCRTO3\\",\\n      \\"xtalkCorrection\\" : [\\n         [ 0, 0, 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 1, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0, 0, 0 ],\\n         [ 0, 0, 0, 0, 0, 0, 0 ]\\n      ]\\n   },\\n   \\"dyeSet\\" : {\\n      \\"baseMap\\" : \\"ACGT\\",\\n      \\"excessNoiseCV\\" : [\\n         0.10000000149011612,\\n         0.10000000149011612,\\n         0.10000000149011612,\\n         0.10000000149011612\\n      ],\\n      \\"ipd2SlowStepRatio\\" : [ 0, 0, 0, 0 ],\\n      \\"ipdMean\\" : [\\n         0.30799999833106995,\\n         0.23399999737739563,\\n         0.23399999737739563,\\n         0.18799999356269836\\n      ],\\n      \\"numAnalog\\" : 4,\\n      \\"pulseWidthMean\\" : [\\n         0.23199999332427979,\\n         0.18500000238418579,\\n         0.1809999942779541,\\n         0.21400000154972076\\n      ],\\n      \\"pw2SlowStepRatio\\" : [\\n         3.2000000476837158,\\n         3.2000000476837158,\\n         3.2000000476837158,\\n         3.2000000476837158\\n      ],\\n      \\"relativeAmp\\" : [ 1, 0.68000000715255737, 0.43000000715255737, 0.27000001072883606 ]\\n   },\\n   \\"runInfo\\" : {\\n      \\"hqrfMethod\\" : \\"N2\\",\\n      \\"instrumentName\\" : \\"traceSimulator\\",\\n      \\"platformId\\" : \\"Kestrel\\"\\n   }\\n}\\n", (esc)
