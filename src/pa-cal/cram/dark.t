  $ TRCOUT=${CRAMTMP}/out.cal.h5
  $ pa-cal --config source.SimInputConfig='{"nRows":32, "nCols":6, "Pedestal":0}' \
  > --cal=Dark --sra=0 --outputFile=${TRCOUT} > /dev/null 2>&1

  $ h5ls ${TRCOUT}/Cal
  Mean                     Dataset {32, 6}
  Variance                 Dataset {32, 6}

  $ h5dump -d "/Cal/Mean[0,0;1,1;20,6]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 0, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 20, 6 );
        BLOCK ( 1, 1 );
        DATA {
        (0,0): 29.3555, 20.416, 56.2246, 47.2715, 38.332, 29.3672,
        (1,0): 40.3027, 29.3555, 20.416, 56.2246, 47.2715, 38.332,
        (2,0): 51.25, 40.3027, 29.3555, 20.416, 56.2246, 47.2715,
        (3,0): 17.418, 51.25, 40.3027, 29.3555, 20.416, 56.2246,
        (4,0): 28.3672, 17.418, 51.25, 40.3027, 29.3555, 20.416,
        (5,0): 39.332, 28.3672, 17.418, 51.25, 40.3027, 29.3555,
        (6,0): 50.2715, 39.332, 28.3672, 17.418, 51.25, 40.3027,
        (7,0): 61.2246, 50.2715, 39.332, 28.3672, 17.418, 51.25,
        (8,0): 27.416, 61.2246, 50.2715, 39.332, 28.3672, 17.418,
        (9,0): 38.3555, 27.416, 61.2246, 50.2715, 39.332, 28.3672,
        (10,0): 49.3027, 38.3555, 27.416, 61.2246, 50.2715, 39.332,
        (11,0): 60.25, 49.3027, 38.3555, 27.416, 61.2246, 50.2715,
        (12,0): 26.418, 60.25, 49.3027, 38.3555, 27.416, 61.2246,
        (13,0): 37.3672, 26.418, 60.25, 49.3027, 38.3555, 27.416,
        (14,0): 48.332, 37.3672, 26.418, 60.25, 49.3027, 38.3555,
        (15,0): 59.2715, 48.332, 37.3672, 26.418, 60.25, 49.3027,
        (16,0): 70.2246, 59.2715, 48.332, 37.3672, 26.418, 60.25,
        (17,0): 36.416, 70.2246, 59.2715, 48.332, 37.3672, 26.418,
        (18,0): 47.3555, 36.416, 70.2246, 59.2715, 48.332, 37.3672,
        (19,0): 58.3027, 47.3555, 36.416, 70.2246, 59.2715, 48.332
        


  $ h5dump -d "/Cal/Variance[12,0;1,1;20,6]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 12, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 20, 6 );
        BLOCK ( 1, 1 );
        DATA {
        (12,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (13,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173,
        (14,0): 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045,
        (15,0): 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946,
        (16,0): 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841,
        (17,0): 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578,
        (18,0): 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767,
        (19,0): 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968,
        (20,0): 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622,
        (21,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (22,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173,
        (23,0): 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045,
        (24,0): 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946,
        (25,0): 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841,
        (26,0): 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578,
        (27,0): 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767,
        (28,0): 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968,
        (29,0): 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622,
        (30,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (31,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173
        

# Positive pedestal
$ TRCOUT=${CRAMTMP}/out.cal.h5
  $ pa-cal --config source.SimInputConfig='{"nRows":32, "nCols":6, "Pedestal":3}' \
  > --cal=Dark --sra=0 --outputFile=${TRCOUT} > /dev/null 2>&1

  $ h5ls ${TRCOUT}/Cal
  Mean                     Dataset {32, 6}
  Variance                 Dataset {32, 6}

  $ h5dump -d "/Cal/Mean[0,0;1,1;20,6]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 0, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 20, 6 );
        BLOCK ( 1, 1 );
        DATA {
        (0,0): 29.3555, 20.416, 56.2246, 47.2715, 38.332, 29.3672,
        (1,0): 40.3027, 29.3555, 20.416, 56.2246, 47.2715, 38.332,
        (2,0): 51.25, 40.3027, 29.3555, 20.416, 56.2246, 47.2715,
        (3,0): 17.418, 51.25, 40.3027, 29.3555, 20.416, 56.2246,
        (4,0): 28.3672, 17.418, 51.25, 40.3027, 29.3555, 20.416,
        (5,0): 39.332, 28.3672, 17.418, 51.25, 40.3027, 29.3555,
        (6,0): 50.2715, 39.332, 28.3672, 17.418, 51.25, 40.3027,
        (7,0): 61.2246, 50.2715, 39.332, 28.3672, 17.418, 51.25,
        (8,0): 27.416, 61.2246, 50.2715, 39.332, 28.3672, 17.418,
        (9,0): 38.3555, 27.416, 61.2246, 50.2715, 39.332, 28.3672,
        (10,0): 49.3027, 38.3555, 27.416, 61.2246, 50.2715, 39.332,
        (11,0): 60.25, 49.3027, 38.3555, 27.416, 61.2246, 50.2715,
        (12,0): 26.418, 60.25, 49.3027, 38.3555, 27.416, 61.2246,
        (13,0): 37.3672, 26.418, 60.25, 49.3027, 38.3555, 27.416,
        (14,0): 48.332, 37.3672, 26.418, 60.25, 49.3027, 38.3555,
        (15,0): 59.2715, 48.332, 37.3672, 26.418, 60.25, 49.3027,
        (16,0): 70.2246, 59.2715, 48.332, 37.3672, 26.418, 60.25,
        (17,0): 36.416, 70.2246, 59.2715, 48.332, 37.3672, 26.418,
        (18,0): 47.3555, 36.416, 70.2246, 59.2715, 48.332, 37.3672,
        (19,0): 58.3027, 47.3555, 36.416, 70.2246, 59.2715, 48.332
        


  $ h5dump -d "/Cal/Variance[12,0;1,1;20,6]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 12, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 20, 6 );
        BLOCK ( 1, 1 );
        DATA {
        (12,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (13,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173,
        (14,0): 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045,
        (15,0): 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946,
        (16,0): 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841,
        (17,0): 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578,
        (18,0): 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767,
        (19,0): 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968,
        (20,0): 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622,
        (21,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (22,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173,
        (23,0): 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045,
        (24,0): 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946,
        (25,0): 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841,
        (26,0): 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578,
        (27,0): 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767,
        (28,0): 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968,
        (29,0): 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622,
        (30,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (31,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173
        

# Negative pedestal, transposed chip
$ TRCOUT=${CRAMTMP}/out.cal.h5
  $ pa-cal --config source.SimInputConfig='{"nRows":6, "nCols":32, "Pedestal":-3}' \
  > --cal=Dark --sra=0 --outputFile=${TRCOUT} > /dev/null 2>&1
  $ h5ls ${TRCOUT}/Cal
  Mean                     Dataset {6, 32}
  Variance                 Dataset {6, 32}

  $ h5dump -d "/Cal/Mean[0,0;1,1;4,28]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 0, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 4, 28 );
        BLOCK ( 1, 1 );
        DATA {
        (0,0): 22.416, 58.2246, 49.2715, 40.332, 31.3672, 22.418, 58.25,
        (0,7): 49.3027, 40.3555, 31.416, 67.2246, 58.2715, 49.332, 40.3672,
        (0,14): 31.418, 67.25, 58.3027, 49.3555, 40.416, 59.2246, 67.2715,
        (0,21): 58.332, 49.3672, 40.418, 76.25, 67.3027, 58.3555, 49.416,
        (1,0): 33.3555, 22.416, 58.2246, 49.2715, 40.332, 31.3672, 22.418,
        (1,7): 58.25, 49.3027, 40.3555, 31.416, 67.2246, 58.2715, 49.332,
        (1,14): 40.3672, 31.418, 67.25, 58.3027, 49.3555, 40.416, 59.2246,
        (1,21): 67.2715, 58.332, 49.3672, 40.418, 76.25, 67.3027, 58.3555,
        (2,0): 44.3027, 33.3555, 22.416, 58.2246, 49.2715, 40.332, 31.3672,
        (2,7): 22.418, 58.25, 49.3027, 40.3555, 31.416, 67.2246, 58.2715,
        (2,14): 49.332, 40.3672, 31.418, 67.25, 58.3027, 49.3555, 40.416,
        (2,21): 59.2246, 67.2715, 58.332, 49.3672, 40.418, 76.25, 67.3027,
        (3,0): 55.25, 44.3027, 33.3555, 22.416, 58.2246, 49.2715, 40.332,
        (3,7): 31.3672, 22.418, 58.25, 49.3027, 40.3555, 31.416, 67.2246,
        (3,14): 58.2715, 49.332, 40.3672, 31.418, 67.25, 58.3027, 49.3555,
        (3,21): 40.416, 59.2246, 67.2715, 58.332, 49.3672, 40.418, 76.25
        


  $ h5dump -d "/Cal/Variance[2,0;1,1;4,28]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 2, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 4, 28 );
        BLOCK ( 1, 1 );
        DATA {
        (2,0): 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767,
        (2,7): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622,
        (2,14): 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173,
        (2,21): 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946,
        (3,0): 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968,
        (3,7): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (3,14): 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045,
        (3,21): 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841,
        (4,0): 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613, 83.6622,
        (4,7): 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173,
        (4,14): 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946,
        (4,21): 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578,
        (5,0): 25.767, 9.33578, 102.841, 65.8946, 37.0045, 16.5173, 124.613,
        (5,7): 83.6622, 50.0968, 25.767, 9.33578, 102.841, 65.8946, 37.0045,
        (5,14): 16.5173, 124.613, 83.6622, 50.0968, 25.767, 9.33578, 102.841,
        (5,21): 65.8946, 37.0045, 16.5173, 124.613, 83.6622, 50.0968, 25.767
        

