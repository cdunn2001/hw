  $ mongo-basecaller --numZmwLanes 1 --config common.lanesPerPool=1 --config basecaller.algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/test.baz > /dev/null

  $ bazviewer --silent -l ${CRAMTMP}/test.baz | tail -n +2 | wc -l
  65

  $ bazviewer --silent -d -n 0 ${CRAMTMP}/test.baz  | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
  $ bazviewer --silent -d -n 63 ${CRAMTMP}/test.baz  | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT

  $ mongo-basecaller --zmwOutputStrideFactor 4 --numZmwLanes 1 --config common.lanesPerPool=1 --config basecaller.algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/test.baz > /dev/null

  $ bazviewer --silent -l ${CRAMTMP}/test.baz | tail -n +2 | wc -l
  65

  $ bazviewer --silent -d -n 4 ${CRAMTMP}/test.baz  | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g'
  ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT

  $ bazviewer --silent -d -n 2 ${CRAMTMP}/test.baz  | tail -n +2
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
