  $ BAZFILE=${CRAMTMP}/designer.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5

  $ smrt-basecaller --config source.TraceReplication='{"numFrames":1024, "numZmwLanes":64,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config algorithm.Metrics.framesPerHFMetricBlock=512 --config layout.lanesPerPool=16   \
  > --config=algorithm.modelEstimationMode=FixedEstimations --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  4097

  $ ${TESTDIR}/verify_designer_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/designer_zmw_0.txt ${CRAMTMP}/exp_designer_zmw_0.txt
  Usage: 
      nwalign [options] seq1 seq2 
      
  
  Options:
    -h, --help            show this help message and exit
    --gap_extend=GAP_EXTEND
                          gap extend penalty (must be integer <= 0)
    --gap_open=GAP_OPEN   gap open penalty (must be integer <= 0)
    --match=MATCH         match score (must be integer > 0)
    --matrix=MATRIX       scoring matrix in ncbi/data/ format,
                          if not specificied, match/mismatch are used
    --server=SERVER       if non-zero integer, a server is started

  $ ${TESTDIR}/verify_designer_zmw.sh $BAZFILE $TRCFILE 10 ${CRAMTMP}/designer_zmw_10.txt ${CRAMTMP}/exp_designer_zmw_10.txt
  Usage: 
      nwalign [options] seq1 seq2 
      
  
  Options:
    -h, --help            show this help message and exit
    --gap_extend=GAP_EXTEND
                          gap extend penalty (must be integer <= 0)
    --gap_open=GAP_OPEN   gap open penalty (must be integer <= 0)
    --match=MATCH         match score (must be integer > 0)
    --matrix=MATRIX       scoring matrix in ncbi/data/ format,
                          if not specificied, match/mismatch are used
    --server=SERVER       if non-zero integer, a server is started

  $ ${TESTDIR}/verify_designer_zmw.sh $BAZFILE $TRCFILE 100 ${CRAMTMP}/designer_zmw_100.txt ${CRAMTMP}/exp_designer_zmw_100.txt
  Usage: 
      nwalign [options] seq1 seq2 
      
  
  Options:
    -h, --help            show this help message and exit
    --gap_extend=GAP_EXTEND
                          gap extend penalty (must be integer <= 0)
    --gap_open=GAP_OPEN   gap open penalty (must be integer <= 0)
    --match=MATCH         match score (must be integer > 0)
    --matrix=MATRIX       scoring matrix in ncbi/data/ format,
                          if not specificied, match/mismatch are used
    --server=SERVER       if non-zero integer, a server is started

  $ ${TESTDIR}/verify_designer_zmw.sh $BAZFILE $TRCFILE 1000 ${CRAMTMP}/designer_zmw_1000.txt ${CRAMTMP}/exp_designer_zmw_1000.txt
  Usage: 
      nwalign [options] seq1 seq2 
      
  
  Options:
    -h, --help            show this help message and exit
    --gap_extend=GAP_EXTEND
                          gap extend penalty (must be integer <= 0)
    --gap_open=GAP_OPEN   gap open penalty (must be integer <= 0)
    --match=MATCH         match score (must be integer > 0)
    --matrix=MATRIX       scoring matrix in ncbi/data/ format,
                          if not specificied, match/mismatch are used
    --server=SERVER       if non-zero integer, a server is started

  $ BAZFILE=${CRAMTMP}/test4.baz
  $ TRCFILE=/pbi/dept/primary/sim/mongo/test4_mongo_acgt_SNR-40.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=1 --config=algorithm.modelEstimationMode=FixedEstimations \
  > --outputbazfile ${BAZFILE} > /dev/null

  $ bazviewer --silent -l ${BAZFILE} | tail -n +1 | wc -l
  257

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 0 ${CRAMTMP}/align_zmw_0.txt ${CRAMTMP}/expected_align_zmw_0.txt
  -CACATTAGCACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTC-CCT-TGATGGTGTCGAAGCAGTGTATTGCTAGTGTGTTGCCCGTTGACGAGTTTGCCGA-TGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCC--TGGTACTGCATACACCCGACGAGTAATAC-
  CCACATTAACACTGTACTAATCGTTCTTGAACGAGAATATCTCAACCCCATTCTCTCCTGTGATGGTGTCGAAGCAGTGTATTGCTACTGTGTTGCCCGTTGACGAGTTTGCCGATTGCTTGCCTGTCATCACGTATGAATACGACTCGGAAAGGGGGCCGTTGGTACTGCATACACCCGACGACTAATACG

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 64 ${CRAMTMP}/align_zmw_64.txt ${CRAMTMP}/expected_align_zmw_64.txt
  -TT-TAATACGCGCCCACACATAGCGTGGCGCGCATCCAGTA-CCAGGGACGGCTCGTCGC-CAATTGCTTCACGAGGGTCAG-CCCAAAACGGTCCATTGGTAAGGTGGTTTCA--ATGCCTCGC-TTTACA-CTCTCACCG-ATGCAAGCTTACCTCATGCGAGATTGTG-A-
  GTTCTAATACGCGCCCACACATAGCCTGGCGCGCATCCAGTACCCAGGGACGGCTGGTCGCGCAATTGCTTCACGAGGGTCAGCCCCAAAACGGTCCATTGGTAAGGTGGTTTCAGTATCCCTCGCTTTTACAGCTCTCACCGTATGCAAGCTTACCTCATGCGAGATTGTGAAC

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 160 ${CRAMTMP}/align_zmw_160.txt ${CRAMTMP}/expected_align_zmw_160.txt
  -ATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACA-GG-TTGC-TAACGCCAGTAAGACCAGG-GGTACCTCTGCCGTCT-CCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTC-TTTGCGT-GCGTGATACAAGGGT-GAGGGTTCTACATGTCCC-T-GG
  AATTATGAGACGGCATAAGTCCAAATCAATGCTCCACCCGAAAAACAGGGTTTGCTTAACGCCAGTAAGACCAGGTGGTACCTCTGCCGTCTGCCTTAGATTGAGGATCGAAGCCCAACACGATCGTCTGGTCTTTTGCGTGGCGTGATACAAGGGTGGAGGGTTCTACATGTCCCTTGGG

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 255 ${CRAMTMP}/align_zmw_255.txt ${CRAMTMP}/expected_align_zmw_255.txt
  --AGTTGGTTTAGGGCGGACCCGCAGGTCGTGC-TATTC-AACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACC-GCAGCTAAATCG-AAATCAGCATGCCT-GCAATATACGACAGTC--CACCGTATCGAGTATGC-CTAGGAACGCACACGGGTCGCACACAGAGCAA-AGGGC
  CAAGTTGGTTTAGGGCGGACCCGCAGGTAGTGCTTATTCGAACGGCTGAGGGACTTCGAGTCCTCCATACTGAGTATATCTGTCTTGCAACCTGCAGCTAAATCGTAAATCAGCATGCCTGGCAATATACCACAGTCGACACCGTATCGAGTATGCTCTAGGAACGCAAACGGGTCGCACACAGAGCAACAGGGC

  $ BAZFILE=${CRAMTMP}/test4_hfmetrics.baz
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":8192, "numZmwLanes":4,"traceFile":"'$TRCFILE'" }' \
  > --config multipleBazFiles=false --config algorithm.Metrics.Method=Host --config layout.lanesPerPool=1                  \
  > --config=algorithm.modelEstimationMode=FixedEstimations --outputbazfile ${BAZFILE} > /dev/null

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 4 ${CRAMTMP}/align_zmw_4.txt ${CRAMTMP}/expected_align_zmw_4.txt
  -CGGTAGAAGTGTACGGCTCTGACATAATGAGCCAGGCCGGTAGGCCGTCTTACACCCCTAGACGAGGTAGGTGTGACAAATAACGTGCCTCACAA-TCCGC-TTGTGTCGAGGTAAGAAGCTAAATAGGCCTGGTCTGTAGAGGCAGACGTTC-GTGCAACGGGACA-TCA
  CCGGTAGAAGTGTACGGCTCTGACATAATGAGCCAGGCCGGTAGGCCGTCTTACACCCCTAGACGAGGTAGGTGTGACAAATAACCTGCCTCACAAGTCCGCTTTGTGTCGAGGTAAGAAGCTAAATAGGCCTGGTCTGTAGAGGCAGACGTTCGGTGCAACGGGACATTCA

  $ ${TESTDIR}/verify_align_zmw.sh $BAZFILE $TRCFILE 18 ${CRAMTMP}/align_zmw_18.txt ${CRAMTMP}/expected_align_zmw_18.txt
  -TCCATGACCTCATAATCATTTA-GTTAAGCACGCCCGGATCAAGTTAGGACCTACACCAGTGGAGGGCAT-CTCATGCCTCGTGCGGGCGTATAGTGGACGTCGAAGGATCTCGGTACTACCAACTAGACAGA-TGAGAAATTGATAGAA-TGACACTCGCTCA-T--
  TTCCATGACCTCATAATCATTTAGGTTAAGCACGACCGGATCAAGTTAGGACCTACACCAGTGGAGGGCATGCTCATGCCTCGTGCGGGCGTATAGTGGACGTCGAAGGATCTCGCTACTACCAACTAGAAAGATTGAGAAATTGATAGAATTGACACTCGCTCATTCG
