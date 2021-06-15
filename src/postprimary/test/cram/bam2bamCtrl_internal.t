  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2>&1

Full program w/ af and cf
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml

in == out
  $ bam2bam -s out_gold.subreadset.xml -o out_inout -j 8 --silent --fasta --fastq
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_inout.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_inout.scraps.bam
  $ diff out_gold.fasta.gz out_inout.fasta.gz
  $ diff out_gold.fastq.gz out_inout.fastq.gz

Check if adapter relabeling works
  $ bam2bam -s out_gold.subreadset.xml -o out_new -j 8 --fasta --fastq --silent
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_new.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_new.scraps.bam
  $ diff out_gold.fasta.gz out_new.fasta.gz
  $ diff out_gold.fastq.gz out_new.fastq.gz

to hq and back
  $ bam2bam -s out_gold.subreadset.xml -o out_hq --hqregion -j 8 --silent
  $ bam2bam -s out_gold.subreadset.xml -o out_back -j 8 --silent
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_back.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_back.scraps.bam
