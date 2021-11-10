  $ TRCFILE=/pbi/dept/primary/sim/mongo/test_designer_mongo_acgt_SNR-40.trc.h5
  $ BAZFILE=tmp.baz

# First a sanity check, that trace re-analysis works on baz files without an AnalysisBatch dataset.
# This is necessary if you want to analyze a Sequel tracefile and preserve the original hole numbers.
# This trace file isn't long enough to actually produce any basecalls without replication, but we can
# still verify that it launches, warns that it can't find the original analysis batch ids, and produces
# a baz file with the expected two lanes
  $ smrt-basecaller --config source.TraceReanalysis='{"traceFile":"'$TRCFILE'", "whitelist":[111,222] }' \
  > --config multipleBazFiles=false --outputbazfile ${BAZFILE} | grep WARN | grep -E "ReAnalysis|Beware"
  * WARN     Running in ReAnalysis mode with a tracefile that does not have an AnalysisBatch dataset, is this a Kestrel tracefile* (glob)

  $ bazviewer --silent -l ${BAZFILE} | grep -v -e "^$" | wc -l
  128
  $ bazviewer --silent -l ${BAZFILE} | grep 111
  111
  $ bazviewer --silent -l ${BAZFILE} | grep 222
  222

# Now do an "original" analysis (via trace replication).  There are 4 pools, so we'll save nothing from the first pool, one
# from the second, two from the third, and three from the last.
  $ ORIGBAZ=original.baz
  $ SAVEDTRACE=saved.trc.h5
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768, "numZmwLanes":64,"traceFile":"'$TRCFILE'","cache":true }' \
  > --config multipleBazFiles=false --config layout.lanesPerPool=16 --outputtrcfile=${SAVEDTRACE} --outputbazfile ${ORIGBAZ}        \
  > --config traceSaver.roi=[[1024,64],[2048,128],[3072,192]] > /dev/null

  $ h5ls -d ${SAVEDTRACE}/TraceData/AnalysisBatch
  AnalysisBatch            Dataset {384}
      Data:
          (0) 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          (23) 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          (45) 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
          (67) 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          (89) 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          (111) 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          (133) 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          (155) 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
          (177) 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3,
          (199) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (221) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (243) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (265) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (287) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (309) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (331) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (353) 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
          (375) 3, 3, 3, 3, 3, 3, 3, 3, 3

# Should be removable after PTSD-787.  For now trace saver doesn't populate the
# ScanData, so we have to frankenstein our own valid file
  $ FIXEDTRACE=fixed.trc.h5
  $ h5copy -p -i ${TRCFILE} -o ${FIXEDTRACE} -s ScanData -d ScanData
  $ h5copy -p -i ${SAVEDTRACE} -o ${FIXEDTRACE} -s Events -d Events
  $ h5copy -p -i ${SAVEDTRACE} -o ${FIXEDTRACE} -s TraceData -d TraceData
  $ mv ${FIXEDTRACE} ${SAVEDTRACE}

# To sanity check things, we're going to *not* do a re-analysis, and we'll process
# the first lane (ZMW 1024).  The hole number wont' be preserved, but we can still
# show that we get a different result from the original, since it will have a
# different DME schedule
  $ BADBAZ=bad.baz
  $ smrt-basecaller --config source.TraceReplication='{"numFrames":32768,"numZmwLanes":1,"traceFile":"'$SAVEDTRACE'","cache":true}' \
  > --config multipleBazFiles=false --outputbazfile ${BADBAZ} > /dev/null
 
  $ bazviewer --silent -d -i 0 ${BADBAZ} > bad.txt
  $ bazviewer --silent -d -i 1024 ${ORIGBAZ} > orig.txt
# "bad" has ~30 extra pulses, because it started basecalling 512 frames earlier
  $ grep Label bad.txt | wc -l
  1696
  $ grep Label orig.txt | wc -l
  1664

# Now do a re-analysis.  We expect the results to be the same as the original
  $ GOODBAZ=good.baz
  $ smrt-basecaller --config source.TraceReanalysis='{"traceFile":"'$SAVEDTRACE'"}' \
  > --config multipleBazFiles=false --outputbazfile ${GOODBAZ} > /dev/null

  $ bazviewer --silent -d -i 0 ${GOODBAZ} > reanalysis.txt
  $ bazviewer --silent -d -i 1024 ${ORIGBAZ} > orig.txt
  $ grep Label reanalysis.txt | wc -l
  1664
  $ grep Label orig.txt | wc -l
  1664
# All that should differ is the zmw index
  $ diff -u reanalysis.txt orig.txt
  --- reanalysis.txt* (glob)
  +++ orig.txt* (glob)
  @@ -11897,7 +11897,7 @@
   \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
   \t\t\t\t} (esc)
   \t\t\t], (esc)
  -\t\t\t"ZMW_ID" : 0, (esc)
  +\t\t\t"ZMW_ID" : 1024, (esc)
   \t\t\t"ZMW_NUMBER" : 1024 (esc)
   \t\t} (esc)
   \t], (esc)
  [1]

  $ bazviewer --silent -d -n 2048 ${GOODBAZ} > reanalysis.txt
  $ bazviewer --silent -d -n 2048 ${ORIGBAZ} > orig.txt
  $ diff -u reanalysis.txt orig.txt
  --- reanalysis.txt* (glob)
  +++ orig.txt* (glob)
  @@ -11673,7 +11673,7 @@
   \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
   \t\t\t\t} (esc)
   \t\t\t], (esc)
  -\t\t\t"ZMW_ID" : 64, (esc)
  +\t\t\t"ZMW_ID" : 2048, (esc)
   \t\t\t"ZMW_NUMBER" : 2048 (esc)
   \t\t} (esc)
   \t], (esc)
  [1]

  $ bazviewer --silent -d -n 3072 ${GOODBAZ} > reanalysis.txt
  $ bazviewer --silent -d -n 3072 ${ORIGBAZ} > orig.txt
  $ diff -u reanalysis.txt orig.txt
  --- reanalysis.txt* (glob)
  +++ orig.txt* (glob)
  @@ -11449,7 +11449,7 @@
   \t\t\t\t\t"TRACE_AUTOCORR" : 0 (esc)
   \t\t\t\t} (esc)
   \t\t\t], (esc)
  -\t\t\t"ZMW_ID" : 192, (esc)
  +\t\t\t"ZMW_ID" : 3072, (esc)
   \t\t\t"ZMW_NUMBER" : 3072 (esc)
   \t\t} (esc)
   \t], (esc)
  [1]
