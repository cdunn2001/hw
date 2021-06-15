Create BAZ files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/singleZmw.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o out_internal.baz   -f $TESTDIR/data/singleZmw.fasta --silent -p > /dev/null 2>&1

#Compare HQ regions w/ expected for the internal run
  $ baz2bam out_internal.baz   -o out_internal --disableAdapterFinding --hqregion -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  Disabling adapter finding
  $ pbvalidate --index out_internal.hqregions.bam 
  $ pbvalidate --index out_internal.scraps.bam 

#Compare HQ regions w/ expected for the production run
  $ baz2bam out_production.baz -o out_production --disableAdapterFinding --hqregion -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  Disabling adapter finding
  $ pbvalidate --index out_production.hqregions.bam 
  $ pbvalidate --index out_production.scraps.bam 

Compare adapters w/ expected for the internal run
  $ baz2bam out_internal.baz -o out_internal.adapters -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ pbvalidate --index out_internal.adapters.subreads.bam 
  $ pbvalidate --index out_internal.adapters.scraps.bam 
  * ZeroLengthWarning: SCRAP record m54006_151205_021320/0/2020_2020 has zero length (glob)
    ZeroLengthWarning)

Compare adapters w/ expected for the production run
  $ baz2bam out_production.baz -o out_production.adapters -Q $TESTDIR/data/singleZmw.fasta -j 8 --silent --metadata=$TESTDIR/data/metadata.xml
  $ pbvalidate --index out_production.adapters.subreads.bam 
  $ pbvalidate --index out_production.adapters.scraps.bam 
  * ZeroLengthWarning: SCRAP record m54006_151205_021320/0/2020_2020 has zero length (glob)
    ZeroLengthWarning)
