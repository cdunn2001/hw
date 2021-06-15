  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1

Full program w/ af and cf
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50

Two control reads should now be in scraps, SubreadSet and BAM counts should match:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_gold.subreadset.xml
  414
  $ samtools flagstat out_gold.subreads.bam | awk 'NR==1{print $1;}'
  414

Full program w/ af and old cf workflow
  $ baz2bam out_gold.baz -o out_new_gold --noSplitControlWorkflow -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50

Two control reads should now be in scraps, SubreadSet and BAM counts should match:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_new_gold.subreadset.xml
  414
  $ samtools flagstat out_new_gold.subreads.bam | awk 'NR==1{print $1;}'
  414

Get a baseline for the total length:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  out_gold.subreadset.xml
  219821

in == out
  $ bam2bam -s out_gold.subreadset.xml -o out_inout -j 8 --silent --fasta --fastq --enableBarcodedAdapters=False --minSubLength=50
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_inout.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_inout.scraps.bam
  $ diff out_gold.fasta.gz out_inout.fasta.gz
  $ diff out_gold.fastq.gz out_inout.fastq.gz

Check if adapter relabeling works
  $ bam2bam -s out_gold.subreadset.xml -o out_new -j 8 --fasta --fastq --silent --enableBarcodedAdapters=False --minSubLength=50
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_new.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_new.scraps.bam
  $ diff out_gold.fasta.gz out_new.fasta.gz
  $ diff out_gold.fastq.gz out_new.fastq.gz

Control relabeling
  $ bam2bam -s out_gold.subreadset.xml -o out_controls -j 8 --silent --enableBarcodedAdapters=False --minSubLength=50
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_controls.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_controls.scraps.bam

to hq and back
  $ bam2bam -s out_gold.subreadset.xml -o out_hq --hqregion -j 8 --silent --enableBarcodedAdapters=False --minSubLength=50
  $ bam2bam -s out_hq.subreadset.xml -o out_back -j 8 --silent --enableBarcodedAdapters=False --minSubLength=50
  $ $TESTDIR/scripts/compare_bams.py out_gold.subreads.bam         out_back.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_gold.scraps.bam           out_back.scraps.bam

Test the number of subreads, total length, with uncalled controls:
  $ baz2bam out_gold.baz --disableControlFiltering -o out_nocf -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50
  Disabling control filtering
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_nocf.subreadset.xml
  474
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  out_nocf.subreadset.xml
  251843

Control unlabeling. NB: Must recall adapters as they were previously ignored in control reads
  $ bam2bam -s out_gold.subreadset.xml -o out_nocontrols --unlabelControls -j 8 --silent --enableBarcodedAdapters=False --minSubLength=50
  Disabling control filtering
  $ $TESTDIR/scripts/compare_bams.py out_nocf.subreads.bam         out_nocontrols.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_nocf.scraps.bam           out_nocontrols.scraps.bam

Test the number of unrolled reads, total length, with uncalled controls:
  $ baz2bam out_gold.baz --disableAdapterFinding --disableControlFiltering -o out_gold -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50
  Disabling adapter finding
  Disabling control filtering
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_gold.subreadset.xml
  50
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  out_gold.subreadset.xml
  287228

Test the number of unrolled reads, total length, with called controls:
  $ baz2bam out_gold.baz --disableAdapterFinding -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50
  Disabling adapter finding
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_gold.subreadset.xml
  43
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  out_gold.subreadset.xml
  249013

Full program w/ af and cf (like above):
  $ baz2bam out_gold.baz -o out_gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50

Make sure that control reads don't end up in the experimental chipstats distributions (SEQ-3197).
- Would be higher with control reads (734 with, 708 without):
  $ xpath -q -e '/PipeStats/InsertReadLenDist/ns:SampleSize/text()'  out_gold.sts.xml
  43
  $ xpath -q -e '/PipeStats/InsertReadLenDist/ns:Sample95thPct/text()'  out_gold.sts.xml
  708
- This should stay the same once we remove the control reads from the other chipstats:
  $ xpath -q -e '/PipeStats/ControlReadLenDist/ns:SampleMean/text()'  out_gold.sts.xml
  4476.29
- Productivity (and a few other dists) should also remain unchanged:
  $ xpath -q -e '/PipeStats/ProdDist'  out_gold.sts.xml
  <ProdDist><ns:NumBins>4</ns:NumBins><ns:BinCounts><ns:BinCount>0</ns:BinCount><ns:BinCount>50</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount></ns:BinCounts><ns:MetricDescription>Productivity</ns:MetricDescription><ns:BinLabels><ns:BinLabel>Empty</ns:BinLabel><ns:BinLabel>Productive</ns:BinLabel><ns:BinLabel>Other</ns:BinLabel><ns:BinLabel>Undefined</ns:BinLabel></ns:BinLabels></ProdDist>

Same but without control calling (similar to before the SEQ-3197 fix)
  $ baz2bam out_gold.baz --disableControlFiltering -o out_nocf -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50
  Disabling control filtering

Make sure that control reads don't end up in the experimental chipstats distributions (SEQ-3197).
  $ xpath -q -e '/PipeStats/InsertReadLenDist/ns:SampleSize/text()'  out_nocf.sts.xml
  50
- Ironically the sample distribution doesn't change much, because we're calling adapters now:
  $ xpath -q -e '/PipeStats/InsertReadLenDist/ns:Sample95thPct/text()'  out_nocf.sts.xml
  708
  $ xpath -q -e '/PipeStats/ControlReadLenDist/ns:SampleMean/text()'  out_nocf.sts.xml
  $ xpath -q -e '/PipeStats/ProdDist'  out_nocf.sts.xml
  <ProdDist><ns:NumBins>4</ns:NumBins><ns:BinCounts><ns:BinCount>0</ns:BinCount><ns:BinCount>50</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount></ns:BinCounts><ns:MetricDescription>Productivity</ns:MetricDescription><ns:BinLabels><ns:BinLabel>Empty</ns:BinLabel><ns:BinLabel>Productive</ns:BinLabel><ns:BinLabel>Other</ns:BinLabel><ns:BinLabel>Undefined</ns:BinLabel></ns:BinLabels></ProdDist>
