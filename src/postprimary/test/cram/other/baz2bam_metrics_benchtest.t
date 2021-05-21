  $ export INDATA=/pbi/dept/primary/testdata/traceDataProps.csv

Use benchtest-t2b to regression test some metrics (indata is ignored for now):
  $ $TESTDIR/scripts/benchtestrunner.sh $INDATA `which baz2bam` .

Grep for failures from the test file:
  $ grep -i failure metrictestresults.xml
  <testsuites errors="0" failures="0" * (glob)
  \t<testsuite errors="0" failures="0" * (glob)
