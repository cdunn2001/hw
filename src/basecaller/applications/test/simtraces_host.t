# a staggered detection model estimation process.  Namely, the first part of
# the input traces will be dropped on the ground while we gather histograms
# for the dme to run, and estimations are staggered so that different lanes
# trigger at different times.

  $ BAZFILE=${CRAMTMP}/designer.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768, "numZmwLanes":64,"traceFile":"'$TRCFILE'","cache":true }' \
  > --config multipleBazFiles=false --config=system.analyzerHardware=Host --config layout.lanesPerPool=16 \
  > --outputbazfile ${BAZFILE} > ${CRAMTMP}/designer_out.txt

  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  4097

# Look at one zmw from each lane.  Due to the dme stagger it is expected that each subsequent test
# will have monotonically fewer bases to align.
  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/replicated_zmw_0.txt ${CRAMTMP}/exp_replicated_zmw_0.txt
  AAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTAC
  AAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 1024 ${CRAMTMP}/replicated_zmw_1024.txt ${CRAMTMP}/exp_replicated_zmw_1024.txt
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 2048 ${CRAMTMP}/replicated_zmw_2048.txt ${CRAMTMP}/exp_replicated_zmw_2048.txt
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 3072 ${CRAMTMP}/replicated_zmw_3072.txt ${CRAMTMP}/exp_replicated_zmw_3072.txt
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 1024 ${CRAMTMP}/replicated_zmw_1024.txt ${CRAMTMP}/exp_replicated_zmw_1024.txt
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 2048 ${CRAMTMP}/replicated_zmw_2048.txt ${CRAMTMP}/exp_replicated_zmw_2048.txt
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 3072 ${CRAMTMP}/replicated_zmw_3072.txt ${CRAMTMP}/exp_replicated_zmw_3072.txt
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT

  $ BAZFILE=${CRAMTMP}/test4.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768, "numZmwLanes":4,"traceFile":"'$TRCFILE'","cache":true }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=system.analyzerHardware=Host  \
  > --outputbazfile ${BAZFILE} > ${CRAMTMP}/test4_out.txt
  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  257

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/replicated_zmw_0.txt ${CRAMTMP}/expected_replicated_zmw_0.txt
  ----------------------------------------------------------------------------------------------------------------------------------------GTCATCACGTATGAATACGACTCGGAAAGGGGGCC--TGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTC-CCT-TGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC--TGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTC-CCT-TGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC--TGGTACTGCATACACCCGACGAGTAATAC-
  AACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 64 ${CRAMTMP}/replicated_zmw_64.txt ${CRAMTMP}/expected_replicated_zmw_64.txt
  -----------------------------------------------------------------------------------------------------------------------------------------------A-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTT-TAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA--ATCCCTCGC-TTTACA-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTT-TAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA--ATCCCTCGC-TTTACA-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTG-A-
  TGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 160 ${CRAMTMP}/replicated_zmw_160.txt ${CRAMTMP}/expected_replicated_zmw_160.txt
  ------------------------------------------------------------------------------------------------------------------------------------------------------------TGATACAAGGGT-GAGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGG-TTGC-TAACGCCAGTAAGACCAGG-GGTACCTC-GCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGT-GCGTGATACAAGGGT-GAGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGG-TTGC-TAACGCCAGTAAGACCAGG-GGTACCTC-GCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGT-GCGTGATACAAGGGT-GAGGGTTCTACATGTCCC-T-GG
  GTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 255 ${CRAMTMP}/replicated_zmw_255.txt ${CRAMTMP}/expected_replicated_zmw_255.txt
  GGTCGCACACAGAGCAAAGGGCCTCGCAAGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCG-AAATCAGCATGCCT-GCAATATA-CACAGTC--CACCGTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGCCTCGCAA-GTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCG-AAATCAGCATGCCT-GCAATATA-CACAGTC--CACCGTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGC
  -------CTC---GC-AA-------G---GTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 64 ${CRAMTMP}/replicated_zmw_64.txt ${CRAMTMP}/expected_replicated_zmw_64.txt
  -----------------------------------------------------------------------------------------------------------------------------------------------A-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTT-TAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA--ATCCCTCGC-TTTACA-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTT-TAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA--ATCCCTCGC-TTTACA-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTG-A-
  TGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 160 ${CRAMTMP}/replicated_zmw_160.txt ${CRAMTMP}/expected_replicated_zmw_160.txt
  ------------------------------------------------------------------------------------------------------------------------------------------------------------TGATACAAGGGT-GAGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGG-TTGC-TAACGCCAGTAAGACCAGG-GGTACCTC-GCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGT-GCGTGATACAAGGGT-GAGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGG-TTGC-TAACGCCAGTAAGACCAGG-GGTACCTC-GCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGT-GCGTGATACAAGGGT-GAGGGTTCTACATGTCCC-T-GG
  GTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 255 ${CRAMTMP}/replicated_zmw_255.txt ${CRAMTMP}/expected_replicated_zmw_255.txt
  GGTCGCACACAGAGCAAAGGGCCTCGCAAGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCG-AAATCAGCATGCCT-GCAATATA-CACAGTC--CACCGTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGCCTCGCAA-GTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCG-AAATCAGCATGCCT-GCAATATA-CACAGTC--CACCGTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGC
  -------CTC---GC-AA-------G---GTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGC

