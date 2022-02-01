  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2>&1
  $ baz2bam out_gold.baz -o out_gold_adapter -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml

Adapter relabeling
  $ bam2bam -s out_gold_adapter.subreadset.xml -o out_new_adapter -j 8 --silent
  $ $TESTDIR/scripts/compare_bams.py out_gold_adapter.subreads.bam out_new_adapter.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold_adapter.scraps.bam   out_new_adapter.scraps.bam

Full program w/ af
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml

Check if adapter relabeling works
  $ bam2bam -s out_gold.subreadset.xml -o out_new -j 8 --fasta --fastq --silent
  $ diff out_gold.fasta.gz out_new.fasta.gz
  $ diff out_gold.fastq.gz out_new.fastq.gz
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam out_new.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam   out_new.scraps.bam

in == out
  $ bam2bam -s out_gold.subreadset.xml -o out_inout -j 8 --silent
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam out_inout.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam   out_inout.scraps.bam

Create fasta and fastq only from subreads
  $ bam2bam -s out_gold.subreadset.xml -o out_stupid_format -j 8 --fasta --fastq --silent
  $ diff out_gold.fasta.gz out_stupid_format.fasta.gz
  $ diff out_gold.fastq.gz out_stupid_format.fastq.gz

to hq and back
  $ bam2bam -s out_gold.subreadset.xml -o out_hq --hqregion -j 8 --silent
  $ bam2bam -s out_hq.subreadset.xml  -o out_back -j 8 --silent
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam out_back.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam   out_back.scraps.bam

Stitch polymerase read and compare to original sequences
  $ bam2bam -s out_gold.subreadset.xml -o out_raw -j 8 --fasta --fastq --polymerase --silent
  $ gunzip out_raw.fasta.gz
  $ awk '{if (NR%2==0) print $1}' out_raw.fasta > out_raw.sequences
  $ diff out_raw.sequences $TESTDIR/data/goldenSubset.sequences
