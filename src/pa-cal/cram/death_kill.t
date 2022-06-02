#
# Test for delay interruption
#
  $ TRCOUT=${CRAMTMP}/out1.trc.h5
  $ LOGOUT=${CRAMTMP}/out1.log
  $ pa-cal --config source.SimInputConfig='{"nRows":320, "nCols":240}' --cal=Dark --sra=0 --outputFile=${TRCOUT} --config source.SimInputConfig.minInputDelaySeconds=15 > /dev/null 2>&1 &
  $ PID=$!
  $ ps -q $PID -o state --no-headers
  [DRS] (re)
  $ sleep 1

  $ /bin/kill -s SIGKILL $PID
  $ sleep 1
  *Killed*pa-cal* (glob)

# Make sure pa-cal stopped
  $ ps -q $PID -o state --no-headers
  [1]

#
# Test for calculation interruption
#
  $ TRCOUT=${CRAMTMP}/out2.trc.h5
  $ LOGOUT=${CRAMTMP}/out2.log
  $ pa-cal --config source.SimInputConfig='{"nRows":3600, "nCols":3600}' --cal=Dark --sra=0 --outputFile=${TRCOUT} > /dev/null 2>&1 &
  $ PID=$!
  $ ps -q $PID -o state --no-headers
  [DRS] (re)
  $ sleep 1

  $ /bin/kill -s SIGKILL $PID
  $ sleep 1
  *Killed*pa-cal* (glob)

# Make sure pa-cal stopped
  $ ps -q $PID -o state --no-headers
  [1]
