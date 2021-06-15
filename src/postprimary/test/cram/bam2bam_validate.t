Generate the test data
  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ baz2bam out_gold.baz -o out_gold_adapter --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_gold_adapter.subreads.bam
  $ pbvalidate --index --max-records=10 out_gold_adapter.scraps.bam

Adapter relabeling
  $ bam2bam -s out_gold_adapter.subreadset.xml -o out_new_adapter --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_new_adapter.subreads.bam
  $ pbvalidate --index --max-records=10 out_new_adapter.scraps.bam

Full program w/ af
  $ baz2bam out_gold.baz -o out_gold --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_gold.subreads.bam
  $ pbvalidate --index --max-records=10 out_gold.scraps.bam

Check if adapter relabeling works
  $ bam2bam -s out_gold.subreadset.xml -o out_new --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_new.subreads.bam
  $ pbvalidate --index --max-records=10 out_new.scraps.bam

in == out
  $ bam2bam -s out_gold.subreadset.xml -o out_inout -j 8 --silent --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_inout.subreads.bam
  $ pbvalidate --index --max-records=10 out_inout.scraps.bam

to hq and back
  $ bam2bam -s out_gold.subreadset.xml -o out_hq --hqregion -j 8 --silent --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_hq.hqregions.bam
  $ pbvalidate --index --max-records=10 out_hq.scraps.bam
  $ bam2bam -s out_hq.subreadset.xml -o out_back --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapter=False
  $ pbvalidate --index --max-records=10 out_back.subreads.bam
  $ pbvalidate --index --max-records=10 out_back.scraps.bam
