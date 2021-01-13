# This cram test is based off simtraces.t, with specific tweaks to accomodate
# a staggered detection model estimation process.  Namely, the first part of
# the input traces will be dropped on the ground while we gather histograms
# for the dme to run, and estimations are staggered so that different lanes
# trigger at different times.

  $ BAZFILE=${CRAMTMP}/designer.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

  $ smrt-basecaller --cache --numZmwLanes 64 --config layout.lanesPerPool=16 --frames=32768 --config=algorithm.modelEstimationMode=DynamicEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer --silent -l ${BAZFILE} | tail -n +2 | wc -l
  4097

# Look at one zmw from each lane.  Due to the dme stagger it is expected that each subsequent test
# will have monotonically fewer bases to align.
  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/replicated_zmw_0.txt ${CRAMTMP}/exp_replicated_zmw_0.txt
  -------------------------TACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTAC
  AAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTACAAAGGGTTTCCCGTACGTACGTACGTACGTAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 1024 ${CRAMTMP}/replicated_zmw_1024.txt ${CRAMTMP}/exp_replicated_zmw_1024.txt
  -------------------------AGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC
  AAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATCAAAGGGTTTCCCGTACGTACGTACGAGTCATC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 2048 ${CRAMTMP}/replicated_zmw_2048.txt ${CRAMTMP}/exp_replicated_zmw_2048.txt
  -------------------------CAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT
  AAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGATAAAGGGTTTCCCGTACGTACGTACGCAGTGAT

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 3072 ${CRAMTMP}/replicated_zmw_3072.txt ${CRAMTMP}/exp_replicated_zmw_3072.txt
  -------------------------CGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT
  AAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGATAAAGGGTTTCCCGTACGTACGACTACGACGAT

  $ BAZFILE=${CRAMTMP}/test4.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --cache --numZmwLanes 4 --config layout.lanesPerPool=1 --frames=32768 --config=algorithm.modelEstimationMode=DynamicEstimations --inputfile ${TRCFILE} --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer --silent -l ${BAZFILE} | tail -n +2 | wc -l
  257

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/replicated_zmw_0.txt ${CRAMTMP}/expected_replicated_zmw_0.txt
  -----------------T---A-TG----AA-------T--AC--G--------------A----CT-C---------G--GAA--A--G----G-----G-G--G-CC-TT---G-G--T----A--------CTG------C------AT---A--C----A-----CC-------C-G-----A--CGA-G--TAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATACGAACAATGGGGTCCACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATAC-
  AACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACGAACAATGGGGTCCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 64 ${CRAMTMP}/replicated_zmw_64.txt ${CRAMTMP}/expected_replicated_zmw_64.txt
  ----A--C--C--TC--AT--GCG---------A---------G-----A-T-------------T-GT---G--A-------AC-----T--G-----AAAC-G--C----------T-G---C-G--T---T-G----TA-A----T-A-C----GC--G-----C-C---C----------AC----A---------C--ATA---G------C---G--TGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGCGCAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCATTATCCCTCGCTTTTACATCTCTCACCGGATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTGTAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGCGCAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCATTATCCCTCGCTTTTACATCTCTCACCGGATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTGTAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGCGCAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCATTATCCCTCGCTTTTACATCTCTCACCGGATGCAAGCTTACCTCATGCGAGATTGTG-A-
  TGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAACTGAAACGCTGCGTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAAC

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 160 ${CRAMTMP}/replicated_zmw_160.txt ${CRAMTMP}/expected_replicated_zmw_160.txt
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------CTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCCTT-GG
  GTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGGGTTCGCGCTTCAATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGG

  $ ${TESTDIR}/verify_replicated_zmw.sh $BAZFILE $TRCFILE 255 ${CRAMTMP}/replicated_zmw_255.txt ${CRAMTMP}/expected_replicated_zmw_255.txt
  ----------------------AGGGCGGACCCGCAGGTCGTGCTTATTCTAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACGACAGTC-ACACCCTATCGAGTATGCTCTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGCCTCGCAACGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGCTTATTCTAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACGACAGTC-ACACCCTATCGAGTATGCTCTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGCCTCGCAACGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTCGTGCTTATTCTAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACGACAGTC-ACACCCTATCGAGTATGCTCTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGC
  CTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGCCTCGCAAGGTGCAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGC

