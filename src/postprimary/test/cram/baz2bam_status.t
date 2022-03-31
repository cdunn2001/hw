Create BAZ files
  $ simbazwriter -o out_production.baz -f $TESTDIR/data/goldenSubset.fasta --silent > /dev/null 2>&1
  $ simbazwriter -o out_internal.baz -f $TESTDIR/data/goldenSubset.fasta --silent -p > /dev/null 2&>1

  $ baz2bam out_production.baz -o out_production --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --statusfd 2 2>ppa_status_prod.txt
  $ baz2bam out_internal.baz -o out_internal --hqregion -j 8 --fasta --silent -Q $TESTDIR/data/goldenSubset.fasta --metadata=$TESTDIR/data/metadata.xml --enableBarcodedAdapters=False --statusfd 2 2>ppa_status_internal.txt

  $ cat ppa_status_prod.txt | grep -m1 Startup
  PA_PPA_STATUS {"counter":0,"counterMax":1,"metrics":{},"ready":false,"stageName":"Startup","stageNumber":0,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  $ cat ppa_status_prod.txt | grep -m1 ParseBazHeaders
  PA_PPA_STATUS {"counter":0,"counterMax":1,"metrics":{},"ready":false,"stageName":"ParseBazHeaders","stageNumber":1,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  $ cat ppa_status_prod.txt | grep -m1 Analyze
  PA_PPA_STATUS {"counter":0,"counterMax":50,"metrics":{},"ready":true,"stageName":"Analyze","stageNumber":2,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":30} (glob)
  $ cat ppa_status_prod.txt | grep -m1 Shutdown
  PA_PPA_STATUS {"counter":0,"counterMax":1,"metrics":{},"ready":false,"stageName":"Shutdown","stageNumber":3,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":30} (glob)
  $ cat ppa_status_internal.txt | grep -m1 Startup
  PA_PPA_STATUS {"counter":0,"counterMax":1,"metrics":{},"ready":false,"stageName":"Startup","stageNumber":0,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  $ cat ppa_status_internal.txt | grep -m1 ParseBazHeaders
  PA_PPA_STATUS {"counter":0,"counterMax":1,"metrics":{},"ready":false,"stageName":"ParseBazHeaders","stageNumber":1,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  $ cat ppa_status_internal.txt | grep -m1 Analyze
  PA_PPA_STATUS {"counter":0,"counterMax":50,"metrics":{},"ready":true,"stageName":"Analyze","stageNumber":2,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":30} (glob)
  $ cat ppa_status_internal.txt | grep -m1 Shutdown
  PA_PPA_STATUS {"counter":0,"counterMax":1,"metrics":{},"ready":false,"stageName":"Shutdown","stageNumber":3,"stageWeights":[1, 3, 95, 1],"state":"progress","timeStamp":"*","timeoutForNextStatus":30} (glob)
