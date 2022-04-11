Create BAZ files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1

  $ baz2bam out_production.baz -o out_production --logoutput mylogfile.txt --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ wc -l mylogfile.txt
  113 mylogfile.txt

  $ baz2bam out_production.baz -o out_production --logoutput mylogfile.txt --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ wc -l mylogfile.txt
  226 mylogfile.txt

  $ baz2bam out_production.baz -o out_production --logfilter WARN --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ wc -l out_production.baz2bam.log
  16 out_production.baz2bam.log
  $ grep -c WARN out_production.baz2bam.log
  16