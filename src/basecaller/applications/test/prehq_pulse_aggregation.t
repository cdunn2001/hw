This cram test uses experimental configurations that are likely to be changed or removed, once
the real preHQ algorithm is brought on line.  The whole test can probably be removed once
maintenance becomes annoying, as long as something else is dropped in its place 

  $ BAZFILE=tmp.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations \
  > --outputbazfile ${BAZFILE} --config=prelimHQ.lookbackSize=1 --config=prelimHQ.enablePreHQ=true                         \
  > --config=prelimHQ.hqThrottleFraction=1.0 > /dev/null

  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  257

  $ bazviewer --silent --summary ${BAZFILE}
  events:47406
  bases:47406
  call A:11618
  call C:11893
  call G:11868
  call T:12027
  {
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/align_zmw_0.txt ${CRAMTMP}/expected_align_zmw_0.txt
  -CACATTACCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTTTGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC-TTGGTACTGCATACACCCGACGAGTAATAC-
  CCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACG

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 64 ${CRAMTMP}/align_zmw_64.txt ${CRAMTMP}/expected_align_zmw_64.txt
  -TTGTAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGCGCAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCATTATCCCTCGCTTTTACATCTCTCACCGGATGCAAGCTTACCTCATGCGAGATTGTG-A-
  GTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAAC

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 160 ${CRAMTMP}/align_zmw_160.txt ${CRAMTMP}/expected_align_zmw_160.txt
  -ATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGTTGCGTGATACAAGGGTGCAGGGTTCTACATGTCCCTT-GG
  AATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGG

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 255 ${CRAMTMP}/align_zmw_255.txt ${CRAMTMP}/expected_align_zmw_255.txt
  -AAGTTGGTTTAGGGCGGACCCGCAGGTCGTGCTTATTCTAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACGACAGTC--CACCGTATCGAGTATGCTCTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGC
  CAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGC

Test again with fewer frames, so that nothing is expected to be marked as an hq region yet
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":4096, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations \
  > --outputbazfile ${BAZFILE} --config=prelimHQ.enablePreHQ=true --config=prelimHQ.hqThrottleFraction=1.0 > /dev/null

  $ bazviewer --silent --summary ${BAZFILE}
  events:0
  bases:0
  {
  \t"TYPE" : "BAZ_OVERVIEW" (esc)
  }
