# This cram test is based off simtraces.t, with specific tweaks to accomodate
# a staggered detection model estimation process.  Namely, the first part of
# the input traces will be dropped on the ground while we gather histograms
# for the dme to run, and estimations are staggered so that different lanes
# trigger at different times.

  $ BAZFILE=${CRAMTMP}/designer.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_new_designer_mongo_acgt_SNR-40.trc.h5

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768, "numZmwLanes":64,"traceFile":"'$TRCFILE'","cache":true }' \
  > --config multipleBazFiles=false --config=system.analyzerHardware=Host --config layout.lanesPerPool=16 \
  > --outputbazfile ${BAZFILE} > ${CRAMTMP}/designer_out.txt

  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  4097

# Look at one zmw from each lane.  Due to the dme stagger it is expected that each subsequent test
# will have monotonically fewer bases to align.
  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/replicated_zmw_0.txt ${CRAMTMP}/exp_replicated_zmw_0.txt
  AAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTAC
  AAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 1024 ${CRAMTMP}/replicated_zmw_1024.txt ${CRAMTMP}/exp_replicated_zmw_1024.txt
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 2048 ${CRAMTMP}/replicated_zmw_2048.txt ${CRAMTMP}/exp_replicated_zmw_2048.txt
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 3072 ${CRAMTMP}/replicated_zmw_3072.txt ${CRAMTMP}/exp_replicated_zmw_3072.txt
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT

  $ BAZFILE=${CRAMTMP}/test4.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_new_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768, "numZmwLanes":4,"traceFile":"'$TRCFILE'","cache":true }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=system.analyzerHardware=Host  \
  > --outputbazfile ${BAZFILE} > ${CRAMTMP}/test4_out.txt
  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  257

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/replicated_zmw_0.txt ${CRAMTMP}/expected_replicated_zmw_0.txt
  ----------------------------------------------------------------------------------------------------------------------------------------GTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATAC-
  AACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 64 ${CRAMTMP}/replicated_zmw_64.txt ${CRAMTMP}/expected_replicated_zmw_64.txt
  --------------------------------------------------------------------------------------------------------------------------------------------T----CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA-TATCCCTCGCTTTTACAGCTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA-TATCCCTCGCTTTTACAGCTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA-TATCCCTCGCTTTTACAGCTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTG-A-
  TGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 160 ${CRAMTMP}/replicated_zmw_160.txt ${CRAMTMP}/expected_replicated_zmw_160.txt
  ----------------------------------------------------------------------------------------------------------------T---GAT---------ACA--A------------------GG-GTG---C--------AGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACA-GG-TTGCTTAACGCCAGTAAGACCAGG-GGTACCTCTGCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACA-GG-TTGCTTAACGCCAGTAAGACCAGG-GGTACCTCTGCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCC-TGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACA-GG-TTGCTTAACGCCAGTAAGACCAGG-GGTACCTCTGCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCC-T-GG
  GTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 255 ${CRAMTMP}/replicated_zmw_255.txt ${CRAMTMP}/expected_replicated_zmw_255.txt
  -----------------GG--T------------C-------GC--A--C--A---C------A----GAG---------C--A--A-A---G----G------GC--C----TCG-----CA--A---C--G-----T------G-C-A-A--GT-T-G-G--T--T-T----------A--GG--G----C-G-G----A----C-C-C-----GC-A---GG--T------------C-----GTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGGAAATCAGCATGCCTGGCAATATACGACAGTC--CACCCTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGCCTCGCAACGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGGAAATCAGCATGCCTGGCAATATACGACAGTC--CACCCTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGCCTCGCAACGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGGAAATCAGCATGCCTGGCAATATACGACAGTC--CACCCTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGC
  CTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGC

