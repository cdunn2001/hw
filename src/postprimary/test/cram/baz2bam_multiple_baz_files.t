  $ testdir=/pbi/dept/primary/sim/kestrel/kestrel-multiple-baz-files

  $ baz2bam $testdir/prod.baz -j8 --fasta --fastq --fullHQ -o single_prod -m $TESTDIR/data/metadata.xml --silent --enableBarcodedAdapters=False
  $ baz2bam $testdir/prod.0.baz $testdir/prod.1.baz $testdir/prod.2.baz -j8 --fasta --fastq --fullHQ -o multiple_prod -m $TESTDIR/data/metadata.xml --silent --enableBarcodedAdapters=False
  $ samtools view single_prod.subreads.bam | head -n 10 > single_prod.subreads.txt
  $ samtools view multiple_prod.subreads.bam | head -n 10 > multiple_prod.subreads.txt
  $ diff single_prod.subreads.txt multiple_prod.subreads.txt
  $ stsTool single_prod.sts.h5 | head -n 10 > single_prod.sts.txt
  $ stsTool multiple_prod.sts.h5 | head -n 10 > multiple_prod.sts.txt
  $ diff single_prod.sts.txt multiple_prod.sts.txt

  $ baz2bam $testdir/internal.baz -j8 --fasta --fastq --fullHQ -o single_internal -m $TESTDIR/data/metadata.xml --silent --enableBarcodedAdapters=False
  $ baz2bam $testdir/internal.0.baz $testdir/internal.1.baz $testdir/internal.2.baz -j8 --fasta --fastq --fullHQ -o multiple_internal -m $TESTDIR/data/metadata.xml --silent --enableBarcodedAdapters=False
  $ samtools view single_internal.subreads.bam | head -n 10 > single_internal.subreads.txt
  $ samtools view multiple_internal.subreads.bam | head -n 10 > multiple_internal.subreads.txt
  $ diff single_internal.subreads.txt multiple_internal.subreads.txt
  $ stsTool single_internal.sts.h5 | head -n 10 > single_internal.sts.txt
  $ stsTool multiple_internal.sts.h5 | head -n 10 > multiple_internal.sts.txt
  $ diff single_internal.sts.txt multiple_internal.sts.txt
