#
# Test for delay interruption
#
  $ pa-cal --config source.SimInputConfig='{"nRows":320, "nCols":240}' --cal=Dark --sra=0 --outputFile=out1.trc.h5 --config source.SimInputConfig.minInputDelaySeconds=15 2>&1 1> stdout1.txt &
  $ PID1=$!
  $ sleep 0.1
# DRS are all valid states for a running process, that might be sleeping or doing IO.  
  $ ps -q $PID1 -o state --no-headers
  [DRS] (re)

  $ false
  [1]

# Send SIGTERM signal. The process should return 143 exit code, which is the code for SIGTERM
  $ /bin/kill -s SIGTERM $PID1
  $ wait $PID1
  [143]
# Find termination log lines
  $ grep 'Terminated' stdout1.txt | sed -E 's/^.*INFO\s+//' | sort | uniq
  Analysis: Exit requested due to "Terminated"=15 signal
  pa-cal/main: Exit requested due to "Terminated"=15 signal

# Make sure pa-cal stopped
  $ ps -q $PID1 -o state --no-headers
  [1]

#
# Test for calculation interruption
#
  $ pa-cal --config source.SimInputConfig='{"nRows":3600, "nCols":3600}' --cal=Dark --sra=0 --outputFile=out2.trc.h5 2>&1 1> stdout2.txt &
  $ PID2=$!
  $ ps -q $PID2 -o state --no-headers
  [DRS] (re)
  $ sleep 0.1

# Send SIGTERM signal
  $ /bin/kill -s SIGTERM $PID2
  $ wait $PID2
  [143]

# Find termination log lines
  $ grep 'Terminated' stdout2.txt | sed -E 's/^.*INFO\s+//' | sort | uniq
  Analysis: Exit requested due to "Terminated"=15 signal
  pa-cal/main: Exit requested due to "Terminated"=15 signal

# Make sure pa-cal stopped
  $ ps -q $PID2 -o state --no-headers
  [1]
