
Kiwi CRF HQRF (now the default)

  $ baz2bam --minSnr 3.75 -o out_crf -m $TESTDIR/data/metadata.xml /pbi/dept/primary/sim/spider/kestrel-test/prod.baz --silent --enableBarcodedAdapters=False

Number of reads and SNRs from subreads.bam

  $ samtools view -c out_crf.subreads.bam
  8000
  $ samtools view out_crf.subreads.bam | head -10 | cut -f19
  sn:B:f,26.3762,49.8892,19.753,30.6053
  sn:B:f,26.4701,50.2023,19.6867,30.2927
  sn:B:f,26.2389,49.8014,19.817,30.6024
  sn:B:f,26.3379,49.5695,19.7395,30.4971
  sn:B:f,26.4103,49.5896,19.6698,30.44
  sn:B:f,26.4084,49.6527,19.7814,30.8051
  sn:B:f,26.3986,49.5301,19.8601,30.6551
  sn:B:f,26.1919,49.4706,19.6261,30.4828
  sn:B:f,26.2389,49.5229,19.7076,30.5484
  sn:B:f,26.2834,49.4511,19.7122,30.436

Number of reads and SNRs from scraps.bam (NB: in Jaguar we were kicking out
parts of reads that are now whole when using the CRF. The only scraps
left were NO-HQ, and those existed in Jaguar as well).

  $ samtools view -c out_crf.scraps.bam
  0
  $ samtools view out_crf.scraps.bam | head -10 | cut -f18
