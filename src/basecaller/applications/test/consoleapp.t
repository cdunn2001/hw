  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

  $ smrt-basecaller --inputfile ${TRCFILE} --numZmwLanes 1 --config multipleBazFiles=false --config prelimHQ.bazBufferChunks=1 --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/test.baz > /dev/null

  $ bazviewer --silent -l ${CRAMTMP}/test.baz | tail -n +1 | wc -l
  65

  $ bazviewer --silent -d -n 0 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA
  $ bazviewer --silent -d -n 63 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA

  $ smrt-basecaller --inputfile ${TRCFILE} --config multipleBazFiles=false --config prelimHQ.bazBufferChunks=1 --config=prelimHQ.zmwOutputStride=4 --numZmwLanes 1 --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/test.baz > /dev/null

  $ bazviewer --silent -l ${CRAMTMP}/test.baz | tail -n +1 | wc -l
  65

  $ bazviewer --silent -d -n 4 ${CRAMTMP}/test.baz  | grep Label | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/[ ,]//g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA

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