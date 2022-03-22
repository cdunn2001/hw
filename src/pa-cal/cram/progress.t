  $ TRCOUT=${CRAMTMP}/out.cal.h5
  $ pa-cal --config source.SimInputConfig='{"nRows":320, "nCols":240, "Pedestal":0}' \
  > --cal=Dark --sra=0 --outputFile=${TRCOUT} --statusfd=1 | grep stageName | uniq -c
        2 \t"stageName" : "StartUp", (esc)
        2 \t"stageName" : "Analyze", (esc)
        2 \t"stageName" : "Shutdown", (esc)