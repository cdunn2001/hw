Create BAZ files
  $ bazwriter -o out_production.baz -f $TESTDIR/data/singleZmw.fasta --silent 2> /dev/null
  $ bazwriter -o out_internal.baz   -f $TESTDIR/data/singleZmw.fasta --silent -p 2> /dev/null

#Compare HQ regions w/ expected for the internal run
  $ baz2bam out_internal.baz   -o out_internal   --hqregion -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ $TESTDIR/scripts/compare_bams.py out_internal.hqregions.bam $TESTDIR/data/singleZmwResults/internal.hqregions.bam
  $ $TESTDIR/scripts/compare_bams.py out_internal.scraps.bam $TESTDIR/data/singleZmwResults/internal.scraps.bam

#Compare HQ regions w/ expected for the production run
  $ baz2bam out_production.baz -o out_production --hqregion -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ $TESTDIR/scripts/compare_bams.py out_production.hqregions.bam $TESTDIR/data/singleZmwResults/production.hqregions.bam
  $ $TESTDIR/scripts/compare_bams.py out_production.scraps.bam $TESTDIR/data/singleZmwResults/production.scraps.bam

Compare adapters w/ expected for the internal run
  $ baz2bam out_internal.baz   -o out_internal.adapters   --adapters $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ $TESTDIR/scripts/compare_bams.py out_internal.adapters.subreads.bam $TESTDIR/data/singleZmwResults/internal.adapters.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_internal.adapters.scraps.bam   $TESTDIR/data/singleZmwResults/internal.adapters.scraps.bam

Compare adapters w/ expected for the production run
  $ baz2bam out_production.baz -o out_production.adapters --adapters $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ $TESTDIR/scripts/compare_bams.py out_production.adapters.subreads.bam $TESTDIR/data/singleZmwResults/production.adapters.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_production.adapters.scraps.bam   $TESTDIR/data/singleZmwResults/production.adapters.scraps.bam

Compare barcodes w/ expected for the internal run
  $ baz2bam out_internal.baz   -o out_internal.barcodes   --adapter $TESTDIR/data/adapter.fasta --barcodes $TESTDIR/data/symmetricBarcodes.fasta --scoreMode symmetric -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ $TESTDIR/scripts/compare_bams.py out_internal.barcodes.subreads.bam $TESTDIR/data/singleZmwResults/internal.barcodes.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_internal.barcodes.scraps.bam   $TESTDIR/data/singleZmwResults/internal.barcodes.scraps.bam

Compare barcodes w/ expected for the production run
  $ baz2bam out_production.baz -o out_production.barcodes --adapter $TESTDIR/data/adapter.fasta --barcodes $TESTDIR/data/symmetricBarcodes.fasta --scoreMode symmetric -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ $TESTDIR/scripts/compare_bams.py out_production.barcodes.subreads.bam $TESTDIR/data/singleZmwResults/production.barcodes.subreads.bam
  $ $TESTDIR/scripts/compare_bams.py out_production.barcodes.scraps.bam   $TESTDIR/data/singleZmwResults/production.barcodes.scraps.bam
