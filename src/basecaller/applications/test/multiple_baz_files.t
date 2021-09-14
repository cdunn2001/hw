  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

# Run in single BAZ file mode.
  $ smrt-basecaller --inputfile ${TRCFILE} --numZmwLanes 4 --config multipleBazFiles=false --config algorithm.Metrics.framesPerHFMetricBlock=512 --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/test.baz > /dev/null
  $ ls ${CRAMTMP}/test.baz
  /tmp/cramtests-*/test.baz (glob)

# Run in (default) multiple BAZ file mode.
  $ smrt-basecaller --inputfile ${TRCFILE} --numZmwLanes 4 --config multipleBazFiles=true --config algorithm.Metrics.framesPerHFMetricBlock=512 --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/multi.baz > /dev/null
  $ ls ${CRAMTMP}/multi*baz
  /tmp/cramtests-*/multi.0.baz (glob)
  /tmp/cramtests-*/multi.1.baz (glob)
  /tmp/cramtests-*/multi.2.baz (glob)
  /tmp/cramtests-*/multi.3.baz (glob)

  $ bazviewer -l --silent ${CRAMTMP}/multi.0.baz | wc -l
  65
  $ bazviewer -l --silent ${CRAMTMP}/multi.1.baz | wc -l
  65
  $ bazviewer -l --silent ${CRAMTMP}/multi.2.baz | wc -l
  65
  $ bazviewer -l --silent ${CRAMTMP}/multi.3.baz | wc -l
  65

# Compare data between the two files.
  $ bazviewer --silent -d -n 0 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > single.txt
  $ bazviewer --silent -d -n 0 ${CRAMTMP}/multi.0.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > multi.txt
  $ diff single.txt multi.txt

  $ bazviewer --silent -d -n 64 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > single.txt
  $ bazviewer --silent -d -n 64 ${CRAMTMP}/multi.1.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > multi.txt
  $ diff single.txt multi.txt

  $ bazviewer --silent -d -n 191 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > single.txt
  $ bazviewer --silent -d -n 191 ${CRAMTMP}/multi.2.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > multi.txt
  $ diff single.txt multi.txt

  $ bazviewer --silent -d -n 224 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > single.txt
  $ bazviewer --silent -d -n 224 ${CRAMTMP}/multi.3.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g' > multi.txt
  $ diff single.txt multi.txt
