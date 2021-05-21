  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2>&1
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --disableControlFiltering
  Disabling control filtering

  $ if [ -s out_gold.subreads.bam ]; then echo "expected"; else echo "exception"; fi
  expected

Create BAM with no subreads, as all of them are filtered by SNR
  $ baz2bam out_gold.baz -o out_snr20 -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --minSnr 2000 --disableControlFiltering
  Disabling control filtering

Recover subreads via lower minSnr (default as in the very first baz2bam call)
  $ bam2bam -s out_snr20.subreadset.xml -o out_snrrecovered -j 8 --silent --minSnr 3.75

Compare recovered BAM files to gold
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_snrrecovered.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_snrrecovered.scraps.bam

