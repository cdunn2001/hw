  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2>&1
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml

  $ if [ -s out_gold.subreads.bam ]; then echo "expected"; else echo "exception"; fi
  expected

Create BAM with no subreads, as all of them are filtered by SL
  $ baz2bam out_gold.baz -o out_slInf -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --minSubLength 200000000

Recover subreads via lower minSubLength (default as in the very first baz2bam call)
  $ bam2bam -s out_slInf.subreadset.xml -o out_slrecovered -j 8 --silent

Compare recovered BAM files to gold
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_slrecovered.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_slrecovered.scraps.bam

