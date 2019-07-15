  $ BAZFILE=${CRAMTMP}/designer.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ mongo-basecaller --numZmwLanes 64 --config common.lanesPerPool=64 --frames=1024 --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer --silent -l ${BAZFILE} | tail -n +2 | wc -l
  4097

  $ ZMW=0
  $ bazviewer --silent -d -n ${ZMW} ${BAZFILE} | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g' | cut -c13- > ${CRAMTMP}/out_${ZMW}.txt
  $ h5dump -d /GroundTruth/Bases -s ${ZMW} -c 1 ${TRCFILE} | grep "(${ZMW})" | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | cut -c12- > ${CRAMTMP}/exp_${ZMW}.txt
  $ diff ${CRAMTMP}/out_${ZMW}.txt ${CRAMTMP}/exp_${ZMW}.txt

  $ ZMW=10
  $ bazviewer --silent -d -n ${ZMW} ${BAZFILE} | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g' | cut -c13- > ${CRAMTMP}/out_${ZMW}.txt
  $ h5dump -d /GroundTruth/Bases -s ${ZMW} -c 1 ${TRCFILE} | grep "(${ZMW})" | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | cut -c12- > ${CRAMTMP}/exp_${ZMW}.txt
  $ diff ${CRAMTMP}/out_${ZMW}.txt ${CRAMTMP}/exp_${ZMW}.txt

  $ ZMW=100
  $ bazviewer --silent -d -n ${ZMW} ${BAZFILE} | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g' | cut -c13- > ${CRAMTMP}/out_${ZMW}.txt
  $ h5dump -d /GroundTruth/Bases -s ${ZMW} -c 1 ${TRCFILE} | grep "(${ZMW})" | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | cut -c12- > ${CRAMTMP}/exp_${ZMW}.txt
  $ diff ${CRAMTMP}/out_${ZMW}.txt ${CRAMTMP}/exp_${ZMW}.txt

  $ ZMW=1000
  $ bazviewer --silent -d -n ${ZMW} ${BAZFILE} | grep READOUT | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | xargs | sed 's/ //g' | cut -c13- > ${CRAMTMP}/out_${ZMW}.txt
  $ h5dump -d /GroundTruth/Bases -s ${ZMW} -c 1 ${TRCFILE} | grep "(${ZMW})" | cut -d':' -f 2 | sed 's/ "//' | sed 's/"//' | cut -c12- > ${CRAMTMP}/exp_${ZMW}.txt
  $ diff ${CRAMTMP}/out_${ZMW}.txt ${CRAMTMP}/exp_${ZMW}.txt
