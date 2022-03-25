#
# Test for delay interruption
#
  $ TRCOUT=${CRAMTMP}/out1.trc.h5
  $ LOGOUT=${CRAMTMP}/out1.log
  $ pa-cal --config source.SimInputConfig='{"nRows":320, "nCols":240}' --cal=Dark --sra=0 --outputFile=${TRCOUT} --config source.SimInputConfig.minInputDelaySeconds=15 > ${LOGOUT} 2>&1 &
  $ PID=$!
  $ ps -q $PID -o state --no-headers
  R

# Send SIGTERM signal
  $ sleep 1
  $ /bin/kill -s SIGTERM $PID

# Find termination log lines
  $ sleep 1
  $ cat ${LOGOUT}  | grep 'Terminated' | sed -E 's/^.*INFO\s+//' | uniq
  pa-cal/main: Exit requested due to "Terminated"=15 signal
  Analysis: Exit requested due to "Terminated"=15 signal

# Make sure pa-cal stopped
  $ sleep 1
  $ ps -q $PID -o state --no-headers
  [1]

#
# Test for calculation interruption
#
  $ TRCOUT=${CRAMTMP}/out2.trc.h5
  $ LOGOUT=${CRAMTMP}/out2.log
  $ pa-cal --config source.SimInputConfig='{"nRows":3600, "nCols":3600}' --cal=Dark --sra=0 --outputFile=${TRCOUT} > ${LOGOUT} 2>&1 &
  $ PID=$!
  $ ps -q $PID -o state --no-headers
  R

# Send SIGTERM signal
  $ sleep 1
  $ /bin/kill -s SIGTERM $PID

# Find termination log lines
  $ sleep 1
  $ cat ${LOGOUT}  | grep 'Terminated' | sed -E 's/^.*INFO\s+//' | uniq
  pa-cal/main: Exit requested due to "Terminated"=15 signal
  Analysis: Exit requested due to "Terminated"=15 signal

# Make sure pa-cal stopped
  $ sleep 1
  $ ps -q $PID -o state --no-headers
  [1]
