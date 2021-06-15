  $ simbazwriter -o out_gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1

Full program w/ af, and cf
  $ baz2bam out_gold.baz -o out_uncorrected -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --disableAdapterCorrection --enableBarcodedAdapters=False --minSubLength=50

Two control reads should now be in scraps, SubreadSet and BAM counts should match:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_uncorrected.subreadset.xml
  409
  $ samtools flagstat out_uncorrected.subreads.bam | awk 'NR==1{print $1;}'
  409

Get a baseline for the total length:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  out_uncorrected.subreadset.xml
  220054

Full program w/ af, and cf and adapter correction.
  $ baz2bam out_gold.baz -o out_corrected -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50

Two control reads should now be in scraps, SubreadSet and BAM counts should match:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()'  out_corrected.subreadset.xml
  414
  $ samtools flagstat out_corrected.subreads.bam | awk 'NR==1{print $1;}'
  414

Get a baseline for the total length:
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  out_corrected.subreadset.xml
  219821

Emit adapter QC metrics:
  $ bam2bam -s out_corrected.subreadset.xml --adpqc -o old_adapter -j 8 --silent --disableAdapterCorrection --enableBarcodedAdapters=False --minSubLength=50
  $ bam2bam -s out_corrected.subreadset.xml --adpqc -o new_adapter -j 8 --silent --enableBarcodedAdapters=False --minSubLength=50
  $ samtools view old_adapter.subreads.bam | cut -f12 > uncorrectedAdapters.txt
  $ samtools view new_adapter.subreads.bam | cut -f12 > correctedAdapters.txt

Check that the adapterqc metrics are actually different (looks like we pick up some low accuracy adapters):
  $ diff -u uncorrectedAdapters.txt correctedAdapters.txt | tail -n +3
  @@ -26,7 +26,8 @@
   ad:Z:0,0.7333,108,1;0,0.814,227,1
   ad:Z:0,0.814,227,1;0,0.8667,245,1
   ad:Z:0,0.8667,245,1;0,0.8043,205,1
  -ad:Z:0,0.8043,205,1;.
  +ad:Z:0,0.8043,205,1;0,0.6875,106,1
  +ad:Z:0,0.6875,106,1;.
   ad:Z:.;0,0.9348,315,1
   ad:Z:0,0.9348,315,1;0,0.9783,305,1
   ad:Z:0,0.9783,305,1;0,0.9773,271,1
  @@ -82,7 +83,8 @@
   ad:Z:.;0,0.8958,242,1
   ad:Z:0,0.8958,242,1;0,0.9149,250,1
   ad:Z:0,0.9149,250,1;0,0.8367,176,1
  -ad:Z:0,0.8367,176,1;0,0.8776,231,1
  +ad:Z:0,0.8367,176,1;0,0.6429,437,1
  +ad:Z:0,0.6429,437,1;0,0.8776,231,1
   ad:Z:0,0.8776,231,1;.
   ad:Z:.;0,0.907,285,1
   ad:Z:0,0.907,285,1;0,0.8776,250,1
  @@ -97,7 +99,8 @@
   ad:Z:0,0.8182,162,1;0,0.7321,191,1
   ad:Z:0,0.7321,191,1;0,0.75,181,1
   ad:Z:0,0.75,181,1;0,0.7193,170,1
  -ad:Z:0,0.7193,170,1;0,0.7556,128,1
  +ad:Z:0,0.7193,170,1;0,0.6731,352,1
  +ad:Z:0,0.6731,352,1;0,0.7556,128,1
   ad:Z:0,0.7556,128,1;.
   ad:Z:.;0,0.7308,225,1
   ad:Z:0,0.7308,225,1;0,0.7292,208,1
  @@ -324,9 +327,11 @@
   ad:Z:0,0.9556,295,1;0,0.9556,271,1
   ad:Z:0,0.9556,271,1;0,0.9375,262,1
   ad:Z:0,0.9375,262,1;.
  -ad:Z:.;0,0.7885,122,1
  +ad:Z:.;0,0.6897,642,1
  +ad:Z:0,0.6897,642,1;0,0.7885,122,1
   ad:Z:0,0.7885,122,1;0,0.8085,211,1
  -ad:Z:0,0.8085,211,1;0,0.8444,186,1
  +ad:Z:0,0.8085,211,1;0,0.6061,389,1
  +ad:Z:0,0.6061,389,1;0,0.8444,186,1
   ad:Z:0,0.8444,186,1;0,0.8478,225,1
   ad:Z:0,0.8478,225,1;0,0.8409,215,1
   ad:Z:0,0.8409,215,1;0,0.814,210,1

Check that the new adapter is roughly halfway through a read:
  $ samtools view old_adapter.subreads.bam | grep 'ad:Z:0,0.8367,176,1;0,0.8776,231,1' | cut -f1
  m54006_151205_021320/13/1720_3009
  $ samtools view new_adapter.subreads.bam | grep 'ad:Z:0,0.8367,176,1;0,0.6429,437,1' | cut -f1
  m54006_151205_021320/13/1720_2353
  $ samtools view new_adapter.subreads.bam | grep 'ad:Z:0,0.6429,437,1;0,0.8776,231,1' | cut -f1
  m54006_151205_021320/13/2395_3009
  $ samtools view old_adapter.subreads.bam | grep 'ad:Z:0,0.8085,211,1;0,0.8444,186,1' | cut -f1
  m54006_151205_021320/40/1801_3113
  $ samtools view new_adapter.subreads.bam | grep 'ad:Z:0,0.8085,211,1;0,0.6061,389,1' | cut -f1
  m54006_151205_021320/40/1801_2535
  $ samtools view new_adapter.subreads.bam | grep 'ad:Z:0,0.6061,389,1;0,0.8444,186,1' | cut -f1
  m54006_151205_021320/40/2568_3113

Full program w/ af, and cf and adapter correction. (Change params to exercise palindrome finder):
  $ baz2bam out_gold.baz --disableSensitiveCorrection -o out_palicorrected --adpqc -Q $TESTDIR/data/goldenSubset.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --minSubLength=50

Emit adapter QC metrics:
  $ samtools view out_palicorrected.subreads.bam | cut -f12 > paliCorrectedAdapters.txt

Check that the adapterqc metrics are actually different (looks like we pick up some low accuracy adapters):
  $ diff -u uncorrectedAdapters.txt paliCorrectedAdapters.txt | tail -n +3
  @@ -82,7 +82,8 @@
   ad:Z:.;0,0.8958,242,1
   ad:Z:0,0.8958,242,1;0,0.9149,250,1
   ad:Z:0,0.9149,250,1;0,0.8367,176,1
  -ad:Z:0,0.8367,176,1;0,0.8776,231,1
  +ad:Z:0,0.8367,176,1;0,0,0,0
  +ad:Z:0,0,0,0;0,0.8776,231,1
   ad:Z:0,0.8776,231,1;.
   ad:Z:.;0,0.907,285,1
   ad:Z:0,0.907,285,1;0,0.8776,250,1

Check that the new adapter produced the appropriate cx flags:
  $ samtools view old_adapter.subreads.bam | grep 'ad:Z:0,0.8367,176,1;0,0.8776,231,1' | cut -f13
  cx:i:3
3 + 128 = 131
  $ samtools view out_palicorrected.subreads.bam | grep 'ad:Z:0,0.8367,176,1;0,0,0,0' | cut -f13
  cx:i:131
3 + 64 = 67
  $ samtools view out_palicorrected.subreads.bam | grep 'ad:Z:0,0,0,0;0,0.8776,231,1' | cut -f13
  cx:i:67
