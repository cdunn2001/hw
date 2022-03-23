  $ TRCOUT=${CRAMTMP}/out.cal.h5
  $ pa-cal --config source.SimInputConfig='{"nRows":320, "nCols":240, "Pedestal":0}' \
  > --cal=Dark --sra=0 --outputFile=${TRCOUT} --statusfd=1 | grep _STATUS
  PA_DARKCAL_STATUS {"counter":0,"counterMax":1,"ready":false,"stageName":"StartUp","stageNumber":0,"stageWeights":[10, 80, 10],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  PA_DARKCAL_STATUS {"counter":1,"counterMax":1,"ready":false,"stageName":"StartUp","stageNumber":0,"stageWeights":[10, 80, 10],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  PA_DARKCAL_STATUS {"counter":0,"counterMax":1,"ready":true,"stageName":"Analyze","stageNumber":1,"stageWeights":[10, 80, 10],"state":"progress","timeStamp":"*","timeoutForNextStatus":600} (glob)
  PA_DARKCAL_STATUS {"counter":1,"counterMax":1,"ready":true,"stageName":"Analyze","stageNumber":1,"stageWeights":[10, 80, 10],"state":"progress","timeStamp":"*","timeoutForNextStatus":600} (glob)
  PA_DARKCAL_STATUS {"counter":0,"counterMax":1,"ready":false,"stageName":"Shutdown","stageNumber":2,"stageWeights":[10, 80, 10],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
  PA_DARKCAL_STATUS {"counter":1,"counterMax":1,"ready":false,"stageName":"Shutdown","stageNumber":2,"stageWeights":[10, 80, 10],"state":"progress","timeStamp":"*","timeoutForNextStatus":300} (glob)
