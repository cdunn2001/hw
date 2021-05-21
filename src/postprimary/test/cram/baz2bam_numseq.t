Generate sts.xml

  $ baz2bam -j 12 -o out -m $TESTDIR/data/metadata.xml --silent /pbi/dept/primary/testdata/sim/softball_snr-50/softball_SNR-50_prod_hf_min_exp_meta.baz

Check for productivity, readtype, and loading

  $ xpath -q -e "/PipeStats/ProdDist" out.sts.xml
  <ProdDist><ns:NumBins>4</ns:NumBins><ns:BinCounts><ns:BinCount>0</ns:BinCount><ns:BinCount>7895</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>105</ns:BinCount></ns:BinCounts><ns:MetricDescription>Productivity</ns:MetricDescription><ns:BinLabels><ns:BinLabel>Empty</ns:BinLabel><ns:BinLabel>Productive</ns:BinLabel><ns:BinLabel>Other</ns:BinLabel><ns:BinLabel>Undefined</ns:BinLabel></ns:BinLabels></ProdDist>

  $ xpath -q -e "/PipeStats/ReadTypeDist" out.sts.xml
  <ReadTypeDist><ns:NumBins>8</ns:NumBins><ns:BinCounts><ns:BinCount>0</ns:BinCount><ns:BinCount>7895</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>105</ns:BinCount></ns:BinCounts><ns:MetricDescription>ReadTypeDist</ns:MetricDescription><ns:BinLabels><ns:BinLabel>Empty</ns:BinLabel><ns:BinLabel>FullHqRead0</ns:BinLabel><ns:BinLabel>FullHqRead1</ns:BinLabel><ns:BinLabel>PartialHqRead0</ns:BinLabel><ns:BinLabel>PartialHqRead1</ns:BinLabel><ns:BinLabel>PartialHqRead2</ns:BinLabel><ns:BinLabel>Indeterminate</ns:BinLabel><ns:BinLabel>Undefined</ns:BinLabel></ns:BinLabels></ReadTypeDist>

  $ xpath -q -e "/PipeStats/LoadingDist" out.sts.xml
  <LoadingDist><ns:NumBins>5</ns:NumBins><ns:BinCounts><ns:BinCount>0</ns:BinCount><ns:BinCount>7895</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>0</ns:BinCount><ns:BinCount>105</ns:BinCount></ns:BinCounts><ns:MetricDescription>Loading</ns:MetricDescription><ns:BinLabels><ns:BinLabel>Empty</ns:BinLabel><ns:BinLabel>Single</ns:BinLabel><ns:BinLabel>Multi</ns:BinLabel><ns:BinLabel>Indeterminate</ns:BinLabel><ns:BinLabel>Undefined</ns:BinLabel></ns:BinLabels></LoadingDist>
