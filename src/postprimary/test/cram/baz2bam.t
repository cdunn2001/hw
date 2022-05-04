Create BAZ files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o out_internal.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2&>1
  $ simbazwriter -o out_spider.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o out_spiderrtal.baz -f $TESTDIR/data/goldenSubset.fasta --silent -r > /dev/null 2>&1


Compare subreads w/ adapters and hqregions
  $ baz2bam out_production.baz -o out_production --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ baz2bam out_internal.baz -o out_internal --adapter $TESTDIR/data/adapter.fasta -Q $TESTDIR/data/goldenSubset.fasta -j 8 --fasta --fastq --silent --minSubLength 10 --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ gunzip out_*.gz
  $ diff out_production.fasta out_internal.fasta
  $ rm *.fast*

Validate the subreadset with xmllint and the XSD schema from master branch.
  $ base=http://bitbucket.pacificbiosciences.com:7990/projects/SEQ/repos/xsd-datamodels/raw
  $ curl --silent $base/PacBioDatasets.xsd?at=refs%2Fheads%2Fmaster > PacBioDatasets.xsd
  $ curl --silent $base/PacBioBaseDataModel.xsd?at=refs%2Fheads%2Fmaster > PacBioBaseDataModel.xsd
  $ curl --silent $base/PacBioCollectionMetadata.xsd?at=refs%2Fheads%2Fmaster > PacBioCollectionMetadata.xsd
  $ curl --silent $base/PacBioSampleInfo.xsd?at=refs%2Fheads%2Fmaster > PacBioSampleInfo.xsd
  $ curl --silent $base/PacBioReagentKit.xsd?at=refs%2Fheads%2Fmaster > PacBioReagentKit.xsd

These should be re-activated once the schemas are updated.
#  $ xmllint --noout --schema PacBioDatasets.xsd out_production.subreadset.xml

Compare hqregion

  $ baz2bam out_production.baz -o out_production --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ baz2bam out_internal.baz -o out_internal --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ gunzip out_*.gz
  $ diff out_production.fasta out_internal.fasta
  $ rm *.fast*

Compare hqregion, doing real region finding on softball data

  $ production_baz=/pbi/dept/primary/sim/kestrel/kestrel-zmw-features.RTO2-fixed/prod.baz
  $ internal_baz=/pbi/dept/primary/sim/kestrel/kestrel-zmw-features.RTO2-fixed/internal.baz

  $ baz2bam ${production_baz} -o out_production --hqregion -j 8 --fasta --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ baz2bam ${internal_baz} -o out_internal --hqregion -j 8 --fasta --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ gunzip out_*.gz
  $ diff out_production.fasta out_internal.fasta
  $ rm *.fast*

Check that all of the files exist
  $ baz2bam ${production_baz} -o out_samedir --hqregion -j 8 --fasta --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ dataset validate out_samedir.subreadset.xml
  out_samedir.subreadset.xml is valid DataSet XML with valid ResourceId references
  $ mkdir subdir_test
  $ baz2bam ${production_baz} -o subdir_test/out_subdir --hqregion -j 8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ dataset validate subdir_test/out_subdir.subreadset.xml
  subdir_test/out_subdir.subreadset.xml is valid DataSet XML with valid ResourceId references

Check that paths are relative
  $ mv subdir_test subdir_relative_test
  $ dataset validate subdir_relative_test/out_subdir.subreadset.xml
  subdir_relative_test/out_subdir.subreadset.xml is valid DataSet XML with valid ResourceId references

Exercise the various HQRF configurations:
  $ baz2bam out_production.baz -o out_sequel_default --hqregion -j8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ grep -c "SEQUEL_CRF_HMM configured" out_sequel_default.baz2bam.log
  1
  $ baz2bam out_production.baz -o out_sequel --hqrfConfig K1 --hqregion -j8 --silent --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False
  $ grep -c "SEQUEL_CRF_HMM configured" out_sequel.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_spider_default --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "SPIDER_CRF_HMM configured" out_spider_default.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_spider_2p0 --hqrfConfig M1 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "SPIDER_CRF_HMM configured" out_spider_2p0.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_spider_3p0 --hqrfConfig N1 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "ZOFFSET_CRF_HMM configured" out_spider_3p0.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_spider_2p0_RTAL --hqrfConfig M2 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "TRAINED_CART_HMM configured" out_spider_2p0_RTAL.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_spider_2p0_RT --hqrfConfig M3 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "TRAINED_CART_CART configured" out_spider_2p0_RT.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_use_baz_al --hqrfConfig M4 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "BAZ_HMM configured" out_use_baz_al.baz2bam.log
  1
  $ baz2bam out_spider.baz -o out_spider_n2 --hqrfConfig N2 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "ZOFFSET_CART_HMM configured" out_spider_n2.baz2bam.log
  1

Test ignoreBazActivityLabels:
  $ baz2bam out_spider.baz -o out_spider_n2_relabel --hqrfConfig N2 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --ignoreBazActivityLabels --enableBarcodedAdapters=False

  $ grep -c "ZOFFSET_CART_HMM configured" out_spider_n2_relabel.baz2bam.log
  1

When we ignore the baz activity labels (which are simulated to be A1) and
compute new ones, the simulated MFMetrics features result in all A0s.
  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' out_spider_n2_relabel.subreadset.xml
  0

Test Spider RTAL BAZ:
  $ baz2bam out_spiderrtal.baz -o out_spiderrtal_n2 --hqrfConfig N2 --hqregion -j8 --silent --metadata=$TESTDIR/data/spider_metadata.xml --enableBarcodedAdapters=False
  $ grep -c "ZOFFSET_CART_HMM configured" out_spiderrtal_n2.baz2bam.log
  1

  $ bazviewer --silent -m -f out_spiderrtal.baz | grep -c "PKMAX_[ACTG]"
  0
  [1]
  $ bazviewer --silent -m -f out_spiderrtal.baz | grep -c "BPZVAR_[ACTG]"
  0
  [1]
  $ bazviewer --silent -m -f out_spiderrtal.baz | grep -c "PKZVAR_[ACTG]"
  0
  [1]

  $ spider_baz_size=$(ls -s out_spider.baz|cut -d' ' -f1)
  $ spiderrtal_baz_size=$(ls -s out_spiderrtal.baz|cut -d' ' -f1)
  $ echo "scale=2; $spiderrtal_baz_size/$spider_baz_size" | bc
  .8[45] (re)

  $ xpath -q -e '/pbds:SubreadSet/pbds:DataSetMetadata/pbds:NumRecords/text()' out_spiderrtal_n2.subreadset.xml
  467
