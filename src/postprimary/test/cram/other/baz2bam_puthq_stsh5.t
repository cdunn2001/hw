  $ module load R/3.1.1

Extract sts.h5

  $ baz2bam -o out -m $TESTDIR/data/metadata.xml --adapters $TESTDIR/data/adapter.fasta /pbi/dept/primary/testdata/SEQ-1005/259200ZMWs.baz --silent 

Run R

  $ $TESTDIR/puthq.r out.sts.h5 out.sts.xml

Extract sts.h5

  $ baz2bam -o out2 -m $TESTDIR/data/metadata.xml --adapters $TESTDIR/data/adapter.fasta /pbi/dept/primary/testdata/SEQ-1005/32K_6HrsZMWs.baz --silent 

Run R

  $ $TESTDIR/puthq.r out2.sts.h5 out2.sts.xml
