INTERNAL BAMS:

Test Wall Start, Wall End consistency with an internal bam:
  $ simbazwriter -o out_internal.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2&>1
  $ baz2bam out_internal.baz -o out_internal -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/check_wswe_startframes.py out_internal.subreads.bam
  $ $TESTDIR/scripts/check_wswe_startframes.py out_internal.scraps.bam

Test that bam2bam without adapter calling on an internal bam preserves ws/we:
  $ bam2bam --disableAdapterFinding -s out_internal.subreadset.xml -o idem_internal -j8 --silent --enableBarcodedAdapters=False
  Disabling adapter finding
  $ $TESTDIR/scripts/check_wswe_startframes.py idem_internal.subreads.bam
  $ $TESTDIR/scripts/check_wswe_startframes.py idem_internal.scraps.bam

Test that bam2bam to polymerase reads on an internal bam preserves ws/we:
  $ bam2bam -s out_internal.subreadset.xml -o poly_internal -j8 --silent --hqregion --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/check_wswe_startframes.py poly_internal.hqregions.bam
  $ $TESTDIR/scripts/check_wswe_startframes.py poly_internal.scraps.bam

Test that bam2bam to zmw reads on an internal bam preserves ws/we:
  $ bam2bam -s out_internal.subreadset.xml -o zmw_internal -j8 --silent --zmw --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/check_wswe_startframes.py zmw_internal.zmws.bam
  $ $TESTDIR/scripts/check_wswe_startframes.py zmw_internal.scraps.bam

Test that bam2bam with adapter calling on a polymerase read internal bam recomputes ws/we from sf:
  $ bam2bam -s poly_internal.subreadset.xml -o poly_adapters_internal -j8 --silent --adapter $TESTDIR/data/adapter.fasta --enableBarcodedAdapters=False
  $ $TESTDIR/scripts/check_wswe_startframes.py poly_adapters_internal.subreads.bam
  $ $TESTDIR/scripts/check_wswe_startframes.py poly_adapters_internal.scraps.bam



PRODUCTION BAMS:

Test that production mode bams have we/ws:
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2&>1
  $ baz2bam out_production.baz -o out_production -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ samtools view out_production.subreads.bam | head -n 1| grep -oE '.{15}ws.{15}'
  \twe:i:31851678\tws:i:28337550\tzm: (esc)
  $ samtools view out_production.scraps.bam | head -n 1| grep -oE '.{15}ws.{15}'
  \twe:i:28289537\tws:i:32\tzm:i:0\tRG (esc)

NB: It would be nice to use check_wswe_startframes.py to compare production and internal bams here,
but the pulses and metrics blocks do not appear to be simulated in a compatible fashion. There are
far more frames occupied in each MF block than there are frames, if you go by pulse widths and ipds.
We will use the other set of simulated data (softball SNR-50) below, which follows more of the rules.

Test that bam2bam without adapter calling on a production bam preserves ws/we:
  $ bam2bam --disableAdapterFinding -s out_production.subreadset.xml -o idem_production -j8 --silent --enableBarcodedAdapters=False
  Disabling adapter finding
  $ samtools view idem_production.subreads.bam | head -n 1| grep -oE '.{15}ws.{15}'
  \twe:i:31851678\tws:i:28337550\tzm: (esc)
  $ samtools view idem_production.scraps.bam | head -n 1| grep -oE '.{15}ws.{15}'
  \twe:i:28289537\tws:i:32\tzm:i:0\tRG (esc)

Test that bam2bam to polymerase reads on a production bam preserves ws/we:
  $ bam2bam -s out_production.subreadset.xml -o poly_production -j8 --silent --hqregion --disableAdapterFinding --enableBarcodedAdapters=False
  Disabling adapter finding
  $ samtools view poly_production.hqregions.bam | head -n 1| grep -oE '.{15}ws.{15}'
  \twe:i:73911562\tws:i:28337550\tzm: (esc)
  $ samtools view poly_production.scraps.bam | head -n 1| grep -oE '.{15}ws.{15}'
  \twe:i:28289537\tws:i:32\tzm:i:0\tRG (esc)

Test that bam2bam to zmw reads on an production bam preserves ws/we:
  $ bam2bam -s out_internal.subreadset.xml -o zmw_production -j8 --silent --zmw --enableBarcodedAdapters=False
  $ samtools view zmw_production.zmws.bam | head -n 1| grep -oE '.{15}ws.{15}'
  we:i:105635271\tws:i:32\tzm:i:0\tRG (esc)
  $ samtools view zmw_production.scraps.bam | wc
        0       0       0

Test that bam2bam with adapter calling on production bam removes ws/we:
  $ bam2bam -s out_production.subreadset.xml -o adapters_production -j8 --silent --enableBarcodedAdapters=False
  $ samtools view adapters_production.subreads.bam | head -n 1| grep -oE '.{15}ws.{15}'
  [1]
  $ samtools view adapters_production.scraps.bam | head -n 1| grep -oE '.{15}ws.{15}'
  [1]


Test on other simulated data:
  $ baz=/pbi/dept/primary/sim/kestrel/kestrel-zmw-features.RTO2/internal.baz
  $ meta=$TESTDIR/data/metadata.xml
  $ bam=out_internal

  $ baz2bam $baz -o $bam -m $meta --silent --whitelistZmwId=0-99 --disableAdapterFinding --enableBarcodedAdapters=False
  Disabling adapter finding

Check that sf and ws/we match:
  $ $TESTDIR/scripts/check_wswe_startframes.py ${bam}.zmws.bam

  $ baz=/pbi/dept/primary/sim/kestrel/kestrel-zmw-features.RTO2/prod.baz
  $ meta=$TESTDIR/data/metadata.xml
  $ bam=out_production

  $ baz2bam $baz -o $bam -m $meta --silent --whitelistZmwId=0-99 --disableAdapterFinding --enableBarcodedAdapters=False
  Disabling adapter finding

Check that the exact (internal) ws/we matches the approximate (production) ws/we values to
within 4096 frames (the current MF block size)
  $ $TESTDIR/scripts/check_wswe_startframes.py out_production.zmws.bam out_internal.zmws.bam

