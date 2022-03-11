Create BAZ files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o out_internal.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2&>1

  $ baz2bam out_production.baz -o out_production --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --statusfd 2 2>ppa_status_prod.txt
  $ baz2bam out_internal.baz -o out_internal --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --statusfd 2 2>ppa_status_internal.txt

  $ cat ppa_status_prod.txt | grep -m1 -B4 -A12 Startup
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 1, (esc)
  \t"ready" : false, (esc)
  \t"stageName" : "Startup", (esc)
  \t"stageNumber" : 0, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 300 (esc)
  }
  $ cat ppa_status_prod.txt | grep -m1 -B4 -A12 ParseBazHeaders
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 1, (esc)
  \t"ready" : false, (esc)
  \t"stageName" : "ParseBazHeaders", (esc)
  \t"stageNumber" : 1, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 300 (esc)
  }
  $ cat ppa_status_prod.txt | grep -m1 -B4 -A12 Analyze
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 50, (esc)
  \t"ready" : true, (esc)
  \t"stageName" : "Analyze", (esc)
  \t"stageNumber" : 2, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 30 (esc)
  }
  $ cat ppa_status_prod.txt | grep -m1 -B4 -A12 Shutdown
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 1, (esc)
  \t"ready" : false, (esc)
  \t"stageName" : "Shutdown", (esc)
  \t"stageNumber" : 3, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 30 (esc)
  }
  $ cat ppa_status_internal.txt | grep -m1 -B4 -A12 Startup
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 1, (esc)
  \t"ready" : false, (esc)
  \t"stageName" : "Startup", (esc)
  \t"stageNumber" : 0, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 300 (esc)
  }
  $ cat ppa_status_internal.txt | grep -m1 -B4 -A12 ParseBazHeaders
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 1, (esc)
  \t"ready" : false, (esc)
  \t"stageName" : "ParseBazHeaders", (esc)
  \t"stageNumber" : 1, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 300 (esc)
  }
  $ cat ppa_status_internal.txt | grep -m1 -B4 -A12 Analyze
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 50, (esc)
  \t"ready" : true, (esc)
  \t"stageName" : "Analyze", (esc)
  \t"stageNumber" : 2, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 30 (esc)
  }
  $ cat ppa_status_internal.txt | grep -m1 -B4 -A12 Shutdown
  PA_PPA_STATUS {
  \t"counter" : 0, (esc)
  \t"counterMax" : 1, (esc)
  \t"ready" : false, (esc)
  \t"stageName" : "Shutdown", (esc)
  \t"stageNumber" : 3, (esc)
  \t"stageWeights" :  (esc)
  \t[ (esc)
  \t\t1, (esc)
  \t\t3, (esc)
  \t\t95, (esc)
  \t\t1 (esc)
  \t], (esc)
  \t"state" : "progress", (esc)
  \t"timeStamp" : "*", (esc) (glob)
  \t"timeoutForNextStatus" : 30 (esc)
  }
