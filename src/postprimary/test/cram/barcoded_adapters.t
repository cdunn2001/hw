  $ simbazwriter -o tc6.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o boar.baz -f $TESTDIR/data/boar_ecoli_6zmws.fasta --silent > /dev/null 2>&1

Standard TC6, +/- enableBarcodedAdapters (look for TC6 vs classify stem)
  $ baz2bam tc6.baz -o tc6_tc6 -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ baz2bam tc6.baz -o tc6_daft -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=True

Check that we can tell that DAFT was used
  $ grep "adapter caller has been configured" tc6_tc6.baz2bam.log | rev | cut -c -65 | rev
  |- An adapter caller has been configured with the standard method
  |- An adapter caller has been configured with the standard method
  |- An adapter caller has been configured with the standard method
  $ grep "adapter caller has been configured" tc6_daft.baz2bam.log | rev | cut -c -65 | rev
   An adapter caller has been configured with the trimToLoop method
  |- An adapter caller has been configured with the standard method
  |- An adapter caller has been configured with the standard method

We still want to get all of the TC6 adapters
  $ samtools flagstat tc6_tc6.subreads.bam | awk 'NR==1{print $1;}'
  407
  $ samtools flagstat tc6_daft.subreads.bam | awk 'NR==1{print $1;}'
  407

And we want them to be in the same place (and not leave too many stems behind)
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  tc6_tc6.subreadset.xml
  219239
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  tc6_daft.subreadset.xml
  219134

Emit adapter QC metrics:
  $ bam2bam -s tc6_tc6.subreadset.xml --adpqc -o tc6_tc6_adpqc -j 8 --silent --enableBarcodedAdapters=False
  $ bam2bam -s tc6_daft.subreadset.xml --adpqc -o tc6_daft_adpqc -j 8 --silent --enableBarcodedAdapters=True
  $ samtools view tc6_tc6_adpqc.subreads.bam | cut -f12 > TC6TC6adpqc.txt
  $ samtools view tc6_daft_adpqc.subreads.bam | cut -f12 > TC6DAFTadpqc.txt

Check that the adapterqc metrics are actually different (looks like we pick up some low accuracy adapters):
  $ diff -u TC6TC6adpqc.txt TC6DAFTadpqc.txt | tail -n +3
  @@ -131,8 +131,8 @@
   ad:Z:0,0.8723,257,1;0,0.7358,251,1
   ad:Z:0,0.7358,251,1;.
   ad:Z:.;0,0.9,296,1
  -ad:Z:0,0.9,296,1;0,0.6522,181,1
  -ad:Z:0,0.6522,181,1;0,0.8605,206,1
  +ad:Z:0,0.9,296,1;0,0.6739,181,1
  +ad:Z:0,0.6739,181,1;0,0.8605,206,1
   ad:Z:0,0.8605,206,1;0,0.9111,285,1
   ad:Z:0,0.9111,285,1;0,0.9773,241,1
   ad:Z:0,0.9773,241,1;0,0.8605,236,1
  @@ -210,8 +210,8 @@
   ad:Z:0,0.9111,251,1;0,0.8478,211,1
   ad:Z:0,0.8478,211,1;0,0.9375,175,1
   ad:Z:0,0.9375,175,1;.
  -ad:Z:.;0,0.7955,255,1
  -ad:Z:0,0.7955,255,1;0,0.9149,227,1
  +ad:Z:.;0,0.8182,255,1
  +ad:Z:0,0.8182,255,1;0,0.9149,227,1
   ad:Z:0,0.9149,227,1;0,0.8824,275,1
   ad:Z:0,0.8824,275,1;0,0.9149,250,1
   ad:Z:0,0.9149,250,1;0,0.8113,225,1
  @@ -275,7 +275,7 @@
   ad:Z:0,0.814,290,1;0,0.9286,291,1
   ad:Z:0,0.9286,291,1;0,0.9111,235,1
   ad:Z:0,0.9111,235,1;0,0.8723,145,1
  -ad:Z:0,0.8723,145,1;.
  +ad:Z:0,0.8723,145,1;0,0.6458,60,1
   ad:Z:.;0,0.8372,207,1
   ad:Z:0,0.8372,207,1;0,0.9149,290,1
   ad:Z:0,0.9149,290,1;0,0.907,295,1
  @@ -393,8 +393,8 @@
   ad:Z:0,0.9348,266,1;0,0.9375,215,1
   ad:Z:0,0.9375,215,1;0,0.9574,231,1
   ad:Z:0,0.9574,231,1;0,0.9149,246,1
  -ad:Z:0,0.9149,246,1;0,0.766,300,1
  -ad:Z:0,0.766,300,1;0,0.8864,231,1
  +ad:Z:0,0.9149,246,1;0,0.7872,300,1
  +ad:Z:0,0.7872,300,1;0,0.8864,231,1
   ad:Z:0,0.8864,231,1;.
   ad:Z:.;0,0.8864,230,1
   ad:Z:0,0.8864,230,1;0,0.9149,280,1

BOAR collection, +/- enableBarcodedAdapters (look for TC6 vs classify stem)
  $ baz2bam boar.baz -o boar_tc6 -Q $TESTDIR/data/boar_ecoli_6zmws.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ baz2bam boar.baz -o boar_daft -Q $TESTDIR/data/boar_ecoli_6zmws.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=True

Confirm that DAFT on is the default:
  $ baz2bam boar.baz -o boar_default -Q $TESTDIR/data/boar_ecoli_6zmws.fasta -j 8 --fasta --fastq --silent --metadata=$TESTDIR/data/metadata.xml

We still want to get all of the TC6 adapters
  $ samtools flagstat boar_tc6.subreads.bam | awk 'NR==1{print $1;}'
  93
  $ samtools flagstat boar_daft.subreads.bam | awk 'NR==1{print $1;}'
  93
  $ samtools flagstat boar_default.subreads.bam | awk 'NR==1{print $1;}'
  93

And we want them to be in the same place (and not leave too many stems behind)
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  boar_tc6.subreadset.xml
  214646
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  boar_daft.subreadset.xml
  216659
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()'  boar_default.subreadset.xml
  216659

Emit adapter QC metrics:
  $ bam2bam -s boar_tc6.subreadset.xml --adpqc -o boar_tc6_adpqc -j 8 --silent --enableBarcodedAdapters=False
  $ bam2bam -s boar_daft.subreadset.xml --adpqc -o boar_daft_adpqc -j 8 --silent --enableBarcodedAdapters=True
  $ bam2bam -s boar_default.subreadset.xml --adpqc -o boar_default_adpqc -j 8 --silent
  $ samtools view boar_tc6_adpqc.subreads.bam | cut -f12 > BOARTC6.txt
  $ samtools view boar_daft_adpqc.subreads.bam | cut -f12 > BOARDAFT.txt
  $ samtools view boar_default_adpqc.subreads.bam | cut -f12 > BOARDEFAULT.txt

Check that the adapterqc metrics are actually different (looks like we pick up some low accuracy adapters):
  $ diff -u BOARTC6.txt BOARDAFT.txt | tail -n +3 | head -n 23
  @@ -1,93 +1,93 @@
  -ad:Z:.;0,0.7674,245,1
  -ad:Z:0,0.7674,245,1;0,0.8,256,1
  -ad:Z:0,0.8,256,1;0,0.7674,291,1
  -ad:Z:0,0.7674,291,1;0,0.7674,242,1
  -ad:Z:0,0.7674,242,1;0,0.8333,295,1
  -ad:Z:0,0.8333,295,1;0,0.7857,250,1
  -ad:Z:0,0.7857,250,1;0,0.8125,192,1
  -ad:Z:0,0.8125,192,1;0,0.8372,266,1
  -ad:Z:0,0.8372,266,1;0,0.8049,251,1
  -ad:Z:0,0.8049,251,1;0,0.8478,261,1
  +ad:Z:.;0,0.8571,260,0
  +ad:Z:0,0.8571,260,0;0,0.931,252,0
  +ad:Z:0,0.931,252,0;0,0.9,296,0
  +ad:Z:0,0.9,296,0;0,0.9,242,0
  +ad:Z:0,0.9,242,0;0,0.9643,305,0
  +ad:Z:0,0.9643,305,0;0,0.9643,241,0
  +ad:Z:0,0.9643,241,0;0,0.9355,217,0
  +ad:Z:0,0.9355,217,0;0,0.9643,271,0
  +ad:Z:0,0.9643,271,0;0,0.9286,255,0
  +ad:Z:0,0.9286,255,0;0,1,260,0
  +ad:Z:.;0,0.8214,531,0
  +ad:Z:0,0.8214,531,0;0,0.9259,266,0
  $ diff -u BOARDAFT.txt BOARDEFAULT.txt | tail -n +3

Use stsTool to check that the hasStem dataset is populated correctly
  $ stsTool tc6_tc6.sts.h5 | head -5 | cut -d, -f8
  LoopOnly
  0
  0
  0
  0
  $ stsTool tc6_daft.sts.h5 | head -5 | cut -d, -f8
  LoopOnly
  0
  0
  0
  0
  $ stsTool boar_tc6.sts.h5 | head -5 | cut -d, -f8
  LoopOnly
  0
  0
  0
  0
  $ stsTool boar_daft.sts.h5 | head -5 | cut -d, -f8
  LoopOnly
  1
  1
  1
  1
