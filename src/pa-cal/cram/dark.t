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
        (0,0): 29.5859, 20.5703, 56.6875, 47.625, 38.6016, 29.5625,
        (1,0): 40.6172, 29.5859, 20.5703, 56.6875, 47.625, 38.6016,
        (2,0): 51.6406, 40.6172, 29.5859, 20.5703, 56.6875, 47.625,
        (3,0): 17.5469, 51.6406, 40.6172, 29.5859, 20.5703, 56.6875,
        (4,0): 28.5625, 17.5469, 51.6406, 40.6172, 29.5859, 20.5703,
        (5,0): 39.6016, 28.5625, 17.5469, 51.6406, 40.6172, 29.5859,
        (6,0): 50.625, 39.6016, 28.5625, 17.5469, 51.6406, 40.6172,
        (7,0): 61.6875, 50.625, 39.6016, 28.5625, 17.5469, 51.6406,
        (8,0): 27.5703, 61.6875, 50.625, 39.6016, 28.5625, 17.5469,
        (9,0): 38.5859, 27.5703, 61.6875, 50.625, 39.6016, 28.5625,
        (10,0): 49.6172, 38.5859, 27.5703, 61.6875, 50.625, 39.6016,
        (11,0): 60.6406, 49.6172, 38.5859, 27.5703, 61.6875, 50.625,
        (12,0): 26.5469, 60.6406, 49.6172, 38.5859, 27.5703, 61.6875,
        (13,0): 37.5625, 26.5469, 60.6406, 49.6172, 38.5859, 27.5703,
        (14,0): 48.6016, 37.5625, 26.5469, 60.6406, 49.6172, 38.5859,
        (15,0): 59.625, 48.6016, 37.5625, 26.5469, 60.6406, 49.6172,
        (16,0): 70.6875, 59.625, 48.6016, 37.5625, 26.5469, 60.6406,
        (17,0): 36.5703, 70.6875, 59.625, 48.6016, 37.5625, 26.5469,
        (18,0): 47.5859, 36.5703, 70.6875, 59.625, 48.6016, 37.5625,
        (19,0): 58.6172, 47.5859, 36.5703, 70.6875, 59.625, 48.6016
        


  $ h5dump -d "/Cal/Variance[12,0;1,1;20,6]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 12, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 20, 6 );
        BLOCK ( 1, 1 );
        DATA {
        (12,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (13,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (14,0): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (15,0): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (16,0): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (17,0): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802,
        (18,0): 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (19,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (20,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (21,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (22,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (23,0): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (24,0): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (25,0): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (26,0): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802,
        (27,0): 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (28,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (29,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (30,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (31,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895
        

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
        (0,0): 29.5859, 20.5703, 56.6875, 47.625, 38.6016, 29.5625,
        (1,0): 40.6172, 29.5859, 20.5703, 56.6875, 47.625, 38.6016,
        (2,0): 51.6406, 40.6172, 29.5859, 20.5703, 56.6875, 47.625,
        (3,0): 17.5469, 51.6406, 40.6172, 29.5859, 20.5703, 56.6875,
        (4,0): 28.5625, 17.5469, 51.6406, 40.6172, 29.5859, 20.5703,
        (5,0): 39.6016, 28.5625, 17.5469, 51.6406, 40.6172, 29.5859,
        (6,0): 50.625, 39.6016, 28.5625, 17.5469, 51.6406, 40.6172,
        (7,0): 61.6875, 50.625, 39.6016, 28.5625, 17.5469, 51.6406,
        (8,0): 27.5703, 61.6875, 50.625, 39.6016, 28.5625, 17.5469,
        (9,0): 38.5859, 27.5703, 61.6875, 50.625, 39.6016, 28.5625,
        (10,0): 49.6172, 38.5859, 27.5703, 61.6875, 50.625, 39.6016,
        (11,0): 60.6406, 49.6172, 38.5859, 27.5703, 61.6875, 50.625,
        (12,0): 26.5469, 60.6406, 49.6172, 38.5859, 27.5703, 61.6875,
        (13,0): 37.5625, 26.5469, 60.6406, 49.6172, 38.5859, 27.5703,
        (14,0): 48.6016, 37.5625, 26.5469, 60.6406, 49.6172, 38.5859,
        (15,0): 59.625, 48.6016, 37.5625, 26.5469, 60.6406, 49.6172,
        (16,0): 70.6875, 59.625, 48.6016, 37.5625, 26.5469, 60.6406,
        (17,0): 36.5703, 70.6875, 59.625, 48.6016, 37.5625, 26.5469,
        (18,0): 47.5859, 36.5703, 70.6875, 59.625, 48.6016, 37.5625,
        (19,0): 58.6172, 47.5859, 36.5703, 70.6875, 59.625, 48.6016
        


  $ h5dump -d "/Cal/Variance[12,0;1,1;20,6]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 12, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 20, 6 );
        BLOCK ( 1, 1 );
        DATA {
        (12,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (13,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (14,0): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (15,0): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (16,0): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (17,0): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802,
        (18,0): 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (19,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (20,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (21,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (22,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (23,0): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (24,0): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (25,0): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (26,0): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802,
        (27,0): 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (28,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (29,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (30,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (31,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895
        

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
        (0,0): 22.5703, 58.6875, 49.625, 40.6016, 31.5625, 22.5469, 58.6406,
        (0,7): 49.6172, 40.5859, 31.5703, 67.6875, 58.625, 49.6016, 40.5625,
        (0,14): 31.5469, 67.6406, 58.6172, 49.5859, 40.5703, 59.6875, 67.625,
        (0,21): 58.6016, 49.5625, 40.5469, 76.6406, 67.6172, 58.5859, 49.5703,
        (1,0): 33.5859, 22.5703, 58.6875, 49.625, 40.6016, 31.5625, 22.5469,
        (1,7): 58.6406, 49.6172, 40.5859, 31.5703, 67.6875, 58.625, 49.6016,
        (1,14): 40.5625, 31.5469, 67.6406, 58.6172, 49.5859, 40.5703, 59.6875,
        (1,21): 67.625, 58.6016, 49.5625, 40.5469, 76.6406, 67.6172, 58.5859,
        (2,0): 44.6172, 33.5859, 22.5703, 58.6875, 49.625, 40.6016, 31.5625,
        (2,7): 22.5469, 58.6406, 49.6172, 40.5859, 31.5703, 67.6875, 58.625,
        (2,14): 49.6016, 40.5625, 31.5469, 67.6406, 58.6172, 49.5859, 40.5703,
        (2,21): 59.6875, 67.625, 58.6016, 49.5625, 40.5469, 76.6406, 67.6172,
        (3,0): 55.6406, 44.6172, 33.5859, 22.5703, 58.6875, 49.625, 40.6016,
        (3,7): 31.5625, 22.5469, 58.6406, 49.6172, 40.5859, 31.5703, 67.6875,
        (3,14): 58.625, 49.6016, 40.5625, 31.5469, 67.6406, 58.6172, 49.5859,
        (3,21): 40.5703, 59.6875, 67.625, 58.6016, 49.5625, 40.5469, 76.6406
        


  $ h5dump -d "/Cal/Variance[2,0;1,1;4,28]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 2, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 4, 28 );
        BLOCK ( 1, 1 );
        DATA {
        (2,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (2,7): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (2,14): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (2,21): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (3,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (3,7): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (3,14): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (3,21): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (4,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (4,7): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (4,14): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (4,21): 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802,
        (5,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (5,7): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (5,14): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (5,21): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535
        

