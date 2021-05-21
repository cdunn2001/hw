Convert bam to fasta
  $ rm -rf $TESTDIR/*.log $TESTDIR/actual.*

  $ bam2fasta -o $TESTDIR/actual -c 1 $TESTDIR/data/test.bam 
  $ gunzip $TESTDIR/actual.fasta.gz
  $ cat $TESTDIR/actual.fasta | tr -d '\n' > $TESTDIR/actual_nnl.fasta
  $ cat $TESTDIR/data/expected.fasta | tr -d '\n' > $TESTDIR/expected_nnl.fasta
  $ diff $TESTDIR/expected_nnl.fasta $TESTDIR/actual_nnl.fasta
  $ rm $TESTDIR/actual.fasta $TESTDIR/expected_nnl.fasta $TESTDIR/actual_nnl.fasta
