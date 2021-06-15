
Assume AdapterLabeler has been built in:

  $ export ADAPTER_LABELER=$TESTDIR/../../../build/src/AdapterLabeler

And we have a very large Fasta of the most trust-worthy sample data to test,
along with the expected output from RS Primary

  $ export FASTA=/mnt/secondary/Share/testdata/postprimary-unittest/adapter/AdapterFinder.perfect.fasta
  $ export EXPECTED=/mnt/secondary/Share/testdata/postprimary-unittest/adapter/AdapterFinder.perfect.csv

Declare and delete any existing output data file:

  $ export TEST_RESULTS=AdapterLabeler.perfect.csv
  $ rm -f $TEST_RESULTS

Run the Adapter Finder on the large fasta and pipe the results to the specified output: 

  $ $ADAPTER_LABELER $FASTA > $TEST_RESULTS

Compare the 

  $ python $TESTDIR/compare_adps.py $EXPECTED $TEST_RESULTS
  extra 0.000248427724864
  match 0.999867945395
  missing 0.000132054604579
  overlap 0.00702365428105
  perfect 0.992844291114
