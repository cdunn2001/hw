Create BAZ files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o out_internal.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2&>1

  $ baz2bam out_production.baz -o out_production --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --statusfd 2 2>ppa_status_prod.txt
  $ baz2bam out_internal.baz -o out_internal --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --statusfd 2 2>ppa_status_internal.txt

  $ cat ppa_status_prod.txt | grep Startup
  \t"stageName" : "Startup", (esc)
  \t"stageName" : "Startup", (esc)
  $ cat ppa_status_internal.txt | grep Shutdown
  \t"stageName" : "Shutdown", (esc)
  \t"stageName" : "Shutdown", (esc)
  \t"stageName" : "Shutdown", (esc)
