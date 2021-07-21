Generate sts.xml and sts.h5

  $ bazFile="/pbi/dept/primary/sim/spider/kestrel-test/prod.baz"
  $ outPrefix="MetricsTest"
  $ baz2bam -o ${outPrefix} -m $TESTDIR/data/metadata.xml ${bazFile} --silent --enableBarcodedAdapters=False

#Compare the sts.h5 to the sts.xml
#
#  $ stschecker --quiet ${outPrefix}.sts.xml ${outPrefix}.sts.h5

Check for baseline sigma

  $ xpath -q -e "/PipeStats/BaselineStdDist[@Channel=\"A\"]/ns:MinBinValue/text()" ${outPrefix}.sts.xml 
  6.34766
  $ xpath -q -e "/PipeStats/BaselineStdDist[@Channel=\"C\"]/ns:MinBinValue/text()" ${outPrefix}.sts.xml 
  6.34766
  $ xpath -q -e "/PipeStats/BaselineStdDist[@Channel=\"G\"]/ns:MinBinValue/text()" ${outPrefix}.sts.xml 
  8.59375
  $ xpath -q -e "/PipeStats/BaselineStdDist[@Channel=\"T\"]/ns:MinBinValue/text()" ${outPrefix}.sts.xml 
  8.59375

SEQ-1660 Check for snr filter and dme failed
This test was neutered when block SNR exclusion was added to HQRF. It may soon
return to prominence due to HQRF churn.

  $ xpath -q -e "/PipeStats/NumFailedSnrFilterZmws/text()" ${outPrefix}.sts.xml
  0

  $ xpath -q -e "/PipeStats/NumFailedDmeZmws/text()" ${outPrefix}.sts.xml
  0

  $ xpath -q -e "/PipeStats/SequencingUmy/text()" ${outPrefix}.sts.xml
  9179509

Check for N50 stats

  $ xpath -q -e "/PipeStats/ReadLenDist/ns:SampleN50/text()" ${outPrefix}.sts.xml
  1151
  $ xpath -q -e "/PipeStats/InsertReadLenDist/ns:SampleN50/text()" ${outPrefix}.sts.xml
  1151
  $ python -m pbreports.report.filter_stats_xml ${outPrefix}.subreadset.xml report.json >& /dev/null
  $ jq '.attributes[3].value' report.json
  1151
  $ jq '.attributes[7].value' report.json
  1151
