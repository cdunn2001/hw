
Run with minSnr from metadata.xml
  $ bazFile="/pbi/dept/primary/sim/kestrel/kestrel-zmw-features.RTO2-fixed/prod.baz"
  $ baz2bam --metadata $TESTDIR/data/testcfg.metadata.xml -o default --whitelistZmwId=0-199 ${bazFile} --silent > /dev/null 2>&1
  $ grep PPAAlgoConfig default.baz2bam.log | awk -F'=' '{print $2}' | jq '.inputFilter.minSnr'
  1.5

Run with command-line argument
  $ baz2bam --minSnr 3.5 --metadata $TESTDIR/data/testcfg.metadata.xml -o minSnr_cmdline --whitelistZmwId=0-199 ${bazFile} --silent > /dev/null 2>&1
  $ grep PPAAlgoConfig minSnr_cmdline.baz2bam.log | awk -F'=' '{print $2}' | jq '.inputFilter.minSnr'
  3.5

Run with no minSnr in metadata.xml, default for platform should be used
  $ baz2bam --metadata $TESTDIR/data/testcfg_nominSnr.metadata.xml -o noSnrmd --whitelistZmwId=0-199 ${bazFile} --silent > /dev/null 2>&1
  $ grep PPAAlgoConfig noSnrmd.baz2bam.log | awk -F'=' '{print $2}' | jq '.inputFilter.minSnr'
  2
