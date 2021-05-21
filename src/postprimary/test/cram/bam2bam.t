  $ simbazwriter -o gold.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ baz2bam gold.baz -o gold_adapter -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --disableControlFiltering --enableBarcodedAdapters=False
  Disabling control filtering

Argument parsing
  $ baz2bam gold.baz -o sequel  -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --disableControlFiltering --enableBarcodedAdapters=False
  Disabling control filtering
  $ baz2bam gold.baz -o spider -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/spider_metadata.xml --disableControlFiltering --enableBarcodedAdapters=False
  Disabling control filtering
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbmeta:Collections/pbmeta:CollectionMetadata/pbmeta:PPAConfig/text()' sequel.subreadset.xml | jq '.inputFilter'
  {
    "minSnr": 3.75
  }
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbmeta:Collections/pbmeta:CollectionMetadata/pbmeta:PPAConfig/text()' spider.subreadset.xml | jq '.inputFilter'
  {
    "minSnr": 2
  }
  $ bam2bam -s spider.subreadset.xml -o spider_snr -j 8 --silent --enableBarcodedAdapters=False
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbmeta:Collections/pbmeta:CollectionMetadata/pbmeta:PPAConfig/text()' spider_snr.subreadset.xml | jq '.inputFilter'
  {
    "minSnr": 2
  }
  $ recalladapters -s spider.subreadset.xml -o spider_snr -j 8 --silent --enableBarcodedAdapters=False
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbmeta:Collections/pbmeta:CollectionMetadata/pbmeta:PPAConfig/text()' spider_snr.subreadset.xml | jq '.inputFilter'
  {
    "minSnr": 2
  }
  $ bam2bam -s spider.subreadset.xml -o snr_mod -j 8 --silent --enableBarcodedAdapters=False --minSnr 2.34
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbmeta:Collections/pbmeta:CollectionMetadata/pbmeta:PPAConfig/text()' snr_mod.subreadset.xml | jq '.inputFilter'
  {
    "minSnr": 2.3399999141693115
  }
  $ bam2bam -s snr_mod.subreadset.xml -o snr_postmod -j 8 --silent --enableBarcodedAdapters=False
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbmeta:Collections/pbmeta:CollectionMetadata/pbmeta:PPAConfig/text()' snr_postmod.subreadset.xml | jq '.inputFilter'
  {
    "minSnr": 2.3399999141693115
  }

Adapter relabeling
  $ bam2bam -s gold_adapter.subreadset.xml -o new_adapter -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold_adapter.subreads.bam new_adapter.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold_adapter.scraps.bam   new_adapter.scraps.bam
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' new_adapter.subreadset.xml
  465
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()' new_adapter.subreadset.xml
  251077

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
  $ bam2bam -s gold_adapter.subreadset.xml --adpqc -o new_adapter_all -j 8 --silent --minHardAccuracy=0.4 --minSoftAccuracy=0.4 --minFlankingScore=-1 --disableAdapterCorrection --enableBarcodedAdapters=False --enableBarcodedAdapters=False
  $ samtools view new_adapter.subreads.bam | grep zm:i:4194370 | cut -f12 > new_adapter_metrics.txt
  $ samtools view new_adapter_all.subreads.bam | grep zm:i:4194370 | cut -f12 > new_adapter_metrics_all.txt
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
  $ baz2bam gold.baz -o gold -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False

HQ regions
  $ baz2bam gold.baz -o goldhq --hqregion -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False

Check if adapter relabeling works
  $ bam2bam -s gold.subreadset.xml -o new --adapter $TESTDIR/data/adapter.fasta -j 8 --fasta --fastq --silent --enableBarcodedAdapters=False
  $ diff gold.fasta.gz new.fasta.gz
  $ diff gold.fastq.gz new.fastq.gz
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam new.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   new.scraps.bam

in == out
  $ bam2bam -s gold.subreadset.xml -o inout -j 8 --silent --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/compare_bams.py gold.subreads.bam inout.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py gold.scraps.bam   inout.scraps.bam

barcode in == out (we no longer do barcoding, but want to support existing barcodes)
$ bam2bam -s /pbi/dept/primary/testdata/bam2bam_barcoded/barcodes.subreadset.xml -o inout -j 8 --silent --enableBarcodedAdapters=False
$ $TESTDIR/scripts/compare_bams.py /pbi/dept/primary/testdata/bam2bam_barcoded/barcodes.subreads.bam inout.subreads.bam
$ $TESTDIR/scripts/compare_bams.py /pbi/dept/primary/testdata/bam2bam_barcoded/barcodes.scraps.bam inout.scraps.bam

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
  $ bam2bam -s gold.subreadset.xml -o raw -j 8 --fasta --fastq --polymerase --silent --enableBarcodedAdapters=False
  $ gunzip raw.fasta.gz
  $ awk '{if (NR%2==0) print $1}' raw.fasta > raw.sequences
  $ diff raw.sequences $TESTDIR/data/goldenSubset.sequences

Run fullHQ for select ZMWs
  $ samtools view /pbi/dept/primary/testdata/bam2bam_fullHQ/m64004_191019_202707_hn77.subreads.bam | cut -f12
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:1
  $ bam2bam --fullHQ -j 8 -b 4 -o m64004_191019_202707_hn77_fullHQ -s /pbi/dept/primary/testdata/bam2bam_fullHQ/m64004_191019_202707_hn77.subreadset.xml --silent --enableBarcodedAdapters=False
  $ samtools view m64004_191019_202707_hn77_fullHQ.subreads.bam | cut -f12
  cx:i:2
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:1
  $ samtools view /pbi/dept/primary/testdata/bam2bam_fullHQ/m64004_191019_202707_hn1033.subreads.bam | cut -f12
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:1
  $ bam2bam --fullHQ -j 8 -b 4 -o m64004_191019_202707_hn1033_fullHQ -s /pbi/dept/primary/testdata/bam2bam_fullHQ/m64004_191019_202707_hn1033.subreadset.xml --silent --enableBarcodedAdapters=False
  $ samtools view m64004_191019_202707_hn1033_fullHQ.subreads.bam | cut -f12
  cx:i:2
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:3
  cx:i:1
