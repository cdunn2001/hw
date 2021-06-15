  $ simbazwriter -o gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ baz2bam gold.baz -o gold_adapter --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False

Adapter relabeling
  $ recalladapters -s gold_adapter.subreadset.xml -o new_adapter --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold_adapter.subreads.bam new_adapter.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold_adapter.scraps.bam   new_adapter.scraps.bam
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' new_adapter.subreadset.xml
  407
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()' new_adapter.subreadset.xml
  219239

Full program w/ af
  $ baz2bam gold.baz -o gold --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False

Check if adapter relabeling works
  $ recalladapters -s gold.subreadset.xml -o new --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam new.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   new.scraps.bam

in == out
  $ recalladapters -s gold.subreadset.xml -o inout -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam inout.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   inout.scraps.bam

subreadset
  $ recalladapters -s gold.subreadset.xml -o sub -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam sub.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   sub.scraps.bam

subreadsetception
  $ recalladapters -s sub.subreadset.xml -o sub2 -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam sub2.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   sub2.scraps.bam

subreadset w/ adapter relabeling
  $ recalladapters -s gold.subreadset.xml -o subfull --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam subfull.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   subfull.scraps.bam
