Create BAZ and BAM files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ baz2bam out_production.baz -o out_production --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ samtools view out_production.subreads.bam  | cut -f1 | cut -f2 -d'/' | sort -n | uniq | wc -l
  43
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' out_production.subreadset.xml
  417
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()' out_production.subreadset.xml
  219895
  $ baz2bam out_production.baz -o out_production_2 --zmwOutputStride 2 --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ samtools view out_production_2.subreads.bam  | cut -f1 | cut -f2 -d'/' | sort -n | uniq | wc -l
  23
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' out_production_2.subreadset.xml
  238
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()' out_production_2.subreadset.xml
  123755
  $ baz2bam out_production.baz -o out_production_3 --zmwOutputStride 3 --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ samtools view out_production_3.subreads.bam  | cut -f1 | cut -f2 -d'/' | sort -n | uniq | wc -l
  15
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' out_production_3.subreadset.xml
  151
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:TotalLength/text()' out_production_3.subreadset.xml
  81388

sts.xml and sts.h5 should still be the same
  $ diff out_production.sts.xml out_production_2.sts.xml
  $ diff out_production.sts.xml out_production_3.sts.xml

  $ stsTool out_production.sts.h5 > out.sts.txt
  $ stsTool out_production_2.sts.h5 > out_2.sts.txt
  $ stsTool out_production_3.sts.h5 > out_3.sts.txt
  $ diff out.sts.txt out_2.sts.txt
  $ diff out.sts.txt out_3.sts.txt
