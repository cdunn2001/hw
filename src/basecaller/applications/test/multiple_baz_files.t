  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

# Run in single BAZ file mode.
  $ smrt-basecaller --inputfile ${TRCFILE} --numZmwLanes 4 --config multipleBazFiles=0 --config prelimHQ.bazBufferChunks=1 --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/test.baz > /dev/null

# Run in (default) multiple BAZ file mode.
  $ smrt-basecaller --inputfile ${TRCFILE} --numZmwLanes 4 --config prelimHQ.bazBufferChunks=1 --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations --config algorithm.pulseAccumConfig.Method=HostSimulatedPulses --frames 1024 --outputbazfile ${CRAMTMP}/multi.baz > /dev/null
  $ ls ${CRAMTMP}/multi*baz
  /tmp/cramtests-*/multi.0.baz (glob)
  /tmp/cramtests-*/multi.1.baz (glob)
  /tmp/cramtests-*/multi.2.baz (glob)
  /tmp/cramtests-*/multi.3.baz (glob)

