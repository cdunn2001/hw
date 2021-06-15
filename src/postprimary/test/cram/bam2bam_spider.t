  $ simbazwriter -l Spider_1p0_NTO -o gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ baz2bam gold.baz -o gold_adapter -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/spider_metadata.xml --disableControlFiltering --enableBarcodedAdapters=False
  Disabling control filtering

Adapter relabeling
  $ bam2bam -s gold_adapter.subreadset.xml -o new_adapter --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold_adapter.subreads.bam new_adapter.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold_adapter.scraps.bam   new_adapter.scraps.bam

Emit adapter QC metrics:
  $ bam2bam -s gold_adapter.subreadset.xml --adpqc -o new_adapter -j 8 --silent --disableAdapterCorrection --enableBarcodedAdapters=False
  $ samtools view gold_adapter.subreads.bam | grep "ad:Z"
  [1]
  $ samtools view new_adapter.subreads.bam | cut -f12 | head -n 3
  ad:Z:.;0,0.8958,266,1
  ad:Z:0,0.8958,266,1;0,0.86,251,1
  ad:Z:0,0.86,251,1;0,0.9783,295,1

Emit adapter QC metrics with global flanking alignments:
  $ bam2bam -s gold_adapter.subreadset.xml --adpqc -o new_adapter_global -j 8 --silent --globalAlnFlanking --enableBarcodedAdapters=False
  $ samtools view new_adapter_global.subreads.bam | cut -f12 | head -n 3
  ad:Z:.;0,0.8958,226,1
  ad:Z:0,0.8958,226,1;0,0.86,208,1
  ad:Z:0,0.86,208,1;0,0.9783,267,1

Emit adapter QC metrics with long global flanking alignments:
  $ bam2bam -s gold_adapter.subreadset.xml --adpqc -o new_adapter_global -j 8 --silent --globalAlnFlanking --flankLength 250 --enableBarcodedAdapters=False
  $ samtools view new_adapter_global.subreads.bam | cut -f12 | head -n 3
  ad:Z:.;0,0.8958,819,1
  ad:Z:0,0.8958,819,1;0,0.86,587,1
  ad:Z:0,0.86,587,1;0,0.9783,843,1


Emit adapter QC metrics for all potential adapters. We should see some
additional, generally low quality, adapter calls:
  $ bam2bam -s gold_adapter.subreadset.xml --adpqc -o new_adapter_all -j 8 --silent --minHardAccuracy=0.4 --minSoftAccuracy=0.4 --minFlankingScore=-1 --disableAdapterCorrection --enableBarcodedAdapters=False
  $ samtools view new_adapter.subreads.bam | grep 'm64008_191217_013420/2/' | cut -f12 > new_adapter_metrics.txt
  $ samtools view new_adapter_all.subreads.bam | grep 'm64008_191217_013420/2/' | cut -f12 > new_adapter_metrics_all.txt
  $ cat new_adapter_metrics.txt
  ad:Z:.;0,0.9375,291,1
  ad:Z:0,0.9375,291,1;0,0.9184,285,1
  ad:Z:0,0.9184,285,1;0,0.9783,160,1
  ad:Z:0,0.9783,160,1;0,0.7955,245,1
  ad:Z:0,0.7955,245,1;0,0.7179,142,1
  ad:Z:0,0.7179,142,1;0,0.7818,241,1
  ad:Z:0,0.7818,241,1;0,0.8913,235,1
  ad:Z:0,0.8913,235,1;.
  $ cat new_adapter_metrics_all.txt
  ad:Z:.;0,0.9375,291,1
  ad:Z:0,0.9375,291,1;0,0.9184,285,1
  ad:Z:0,0.5,83,1;0,0.9783,160,1
  ad:Z:0,0.9783,160,1;0,0.7955,245,1
  ad:Z:0,0.7955,245,1;0,0.7179,142,1
  ad:Z:0,0.7179,142,1;0,0.7818,241,1
  ad:Z:0,0.7818,241,1;0,0.8913,235,1
  ad:Z:0,0.8913,235,1;.

Full program w/ af
  $ baz2bam gold.baz -o gold --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False

HQ regions
  $ baz2bam gold.baz -o goldhq --hqregion -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False

Check if adapter relabeling works
  $ bam2bam -s gold.subreadset.xml -o new -j 8 --fasta --fastq --silent --enableBarcodedAdapters=False
  $ diff gold.fasta.gz new.fasta.gz
  $ diff gold.fastq.gz new.fastq.gz
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam new.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   new.scraps.bam

in == out
  $ bam2bam -s gold.subreadset.xml -o inout -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam inout.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   inout.scraps.bam

subreadset
  $ bam2bam -s gold.subreadset.xml -o sub -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam sub.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   sub.scraps.bam

subreadsetception
  $ bam2bam -s sub.subreadset.xml -o sub2 -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam sub2.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   sub2.scraps.bam

subreadset HQ
  $ bam2bam -s goldhq.subreadset.xml -o subhq -j 8 --silent --hqregion --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py goldhq.hqregions.bam subhq.hqregions.bam
  $ $TESTDIR/scripts/compare_bams.py goldhq.scraps.bam   subhq.scraps.bam

subreadsetception HQ
  $ bam2bam -s subhq.subreadset.xml -o subhq2 -j 8 --silent --hqregion --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py goldhq.hqregions.bam subhq2.hqregions.bam
  $ $TESTDIR/scripts/compare_bams.py goldhq.scraps.bam   subhq2.scraps.bam

subreadsetception HQ to subreads
  $ bam2bam -s subhq2.subreadset.xml -o subhq2adapters -j 8 --silent --adapter $TESTDIR/data/adapter.fasta --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam subhq2adapters.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   subhq2adapters.scraps.bam

subreadset w/ adapter relabeling
  $ bam2bam -s gold.subreadset.xml -o subfull --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam subfull.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   subfull.scraps.bam

subreadset HQ w/ adapter relabeling
  $ bam2bam -s goldhq.subreadset.xml -o subhqfull --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam subhqfull.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   subhqfull.scraps.bam

subreadset to HQ
  $ bam2bam -s gold.subreadset.xml -o subtohq -j 8 --silent --hqregion --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py goldhq.hqregions.bam subtohq.hqregions.bam
  $ $TESTDIR/scripts/compare_bams.py goldhq.scraps.bam   subtohq.scraps.bam

Create fasta and fastq only from subreads
  $ bam2bam -s gold.subreadset.xml -o stupid_format -j 8 --fasta --fastq --silent --enableBarcodedAdapters=False
  $ diff gold.fasta.gz stupid_format.fasta.gz
  $ diff gold.fastq.gz stupid_format.fastq.gz

to hq and back
  $ bam2bam -s gold.subreadset.xml -o hq --hqregion -j 8 --silent --enableBarcodedAdapters=False
  $ bam2bam -s hq.subreadset.xml -o back --adapter $TESTDIR/data/adapter.fasta -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam back.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   back.scraps.bam

Stitch polymerase read and compare to original sequences
  $ bam2bam -s gold.subreadset.xml -o raw -j 8 --fasta --fastq --polymerase --silent --disableAdapterFinding --enableBarcodedAdapters=False
  Disabling adapter finding
  $ gunzip raw.fasta.gz
  $ awk '{if (NR%2==0) print $1}' raw.fasta > raw.sequences
  $ diff raw.sequences $TESTDIR/data/goldenSubset.sequences
