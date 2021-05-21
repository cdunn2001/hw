Generate the BAZ file
  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2>&1

Full program w/ af, and cf
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ pbvalidate --index --max-records=10 out_gold.subreads.bam
  $ pbvalidate --index --max-records=10 out_gold.scraps.bam

in == out
  $ bam2bam -s out_gold.subreadset.xml -o out_inout -j 8 --silent
  $ pbvalidate --index --max-records=10 out_inout.subreads.bam
  $ pbvalidate --index --max-records=10 out_inout.scraps.bam

Check if adapter relabeling works
  $ bam2bam -s out_gold.subreadset.xml -o out_new -j 8 --silent
  $ pbvalidate --index --max-records=10 out_new.subreads.bam
  $ pbvalidate --index --max-records=10 out_new.scraps.bam

to hq and back
  $ bam2bam -s out_gold.subreadset.xml -o out_hq --hqregion -j 8 --silent
  $ pbvalidate --index --max-records=10 out_hq.hqregions.bam
  $ pbvalidate --index --max-records=10 out_hq.scraps.bam
  $ bam2bam -s out_hq.subreadset.xml -o out_back -j 8 --silent
  $ pbvalidate --index --max-records=10 out_back.subreads.bam
  $ pbvalidate --index --max-records=10 out_back.scraps.bam
