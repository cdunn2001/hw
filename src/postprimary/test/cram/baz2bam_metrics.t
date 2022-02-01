Create BAZ files
  $ simbazwriter -p -o twoZmws.baz -f $TESTDIR/data/twoZmwsOneEmpty.fasta --silent > /dev/null 2>&1

Generate sts.xml
  $ baz2bam twoZmws.baz -o twoZmws --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/twoZmwsOneEmpty.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False

Test the productivity sum
  $ $TESTDIR/scripts/sum_productivity.sh twoZmws.sts.xml
  2
