  $ TRCOUT=${CRAMTMP}/out.cal.h5
  $ pa-cal --config source.SimInputConfig='{"nRows":24, "nCols":8, "Pedestal":0}' \
  > --cal=Loading --sra=0 --outputFile=${TRCOUT} > /dev/null 2>&1

  $ h5ls ${TRCOUT}/Loading
  LoadingMean              Dataset {24, 8}
  LoadingVariance          Dataset {24, 8}

  $ h5dump -d "/Loading/LoadingMean[0,0;1,1;16,7]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 0, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 16, 7 );
        BLOCK ( 1, 1 );
        DATA {
        (0,0): 49.6406, 40.6172, 31.5859, 22.5703, 58.6875, 49.625, 40.6016,
        (1,0): 15.5469, 49.6406, 40.6172, 31.5859, 22.5703, 58.6875, 49.625,
        (2,0): 26.5625, 15.5469, 49.6406, 40.6172, 31.5859, 22.5703, 58.6875,
        (3,0): 37.6016, 26.5625, 15.5469, 49.6406, 40.6172, 31.5859, 22.5703,
        (4,0): 48.625, 37.6016, 26.5625, 15.5469, 49.6406, 40.6172, 31.5859,
        (5,0): 59.6875, 48.625, 37.6016, 26.5625, 15.5469, 49.6406, 40.6172,
        (6,0): 25.5703, 59.6875, 48.625, 37.6016, 26.5625, 15.5469, 49.6406,
        (7,0): 36.5859, 25.5703, 59.6875, 48.625, 37.6016, 26.5625, 15.5469,
        (8,0): 47.6172, 36.5859, 25.5703, 59.6875, 48.625, 37.6016, 26.5625,
        (9,0): 58.6406, 47.6172, 36.5859, 25.5703, 59.6875, 48.625, 37.6016,
        (10,0): 24.5469, 58.6406, 47.6172, 36.5859, 25.5703, 59.6875, 48.625,
        (11,0): 35.5625, 24.5469, 58.6406, 47.6172, 36.5859, 25.5703, 59.6875,
        (12,0): 46.6016, 35.5625, 24.5469, 58.6406, 47.6172, 36.5859, 25.5703,
        (13,0): 57.625, 46.6016, 35.5625, 24.5469, 58.6406, 47.6172, 36.5859,
        (14,0): 68.6875, 57.625, 46.6016, 35.5625, 24.5469, 58.6406, 47.6172,
        (15,0): 34.5703, 68.6875, 57.625, 46.6016, 35.5625, 24.5469, 58.6406
        

  $ h5dump -d "/Loading/LoadingVariance[8,0;1,1;16,7]" ${TRCOUT} | awk -v n=2 -v RS='}' 'NR==n{ print }'
  
     SUBSET {
        START ( 8, 0 );
        STRIDE ( 1, 1 );
        COUNT ( 16, 7 );
        BLOCK ( 1, 1 );
        DATA {
        (8,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (9,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (10,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (11,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (12,0): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (13,0): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (14,0): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806,
        (15,0): 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878,
        (16,0): 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535, 8.10802,
        (17,0): 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344, 22.1535,
        (18,0): 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268, 42.9344,
        (19,0): 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807, 71.8268,
        (20,0): 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895, 107.807,
        (21,0): 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713, 14.0895,
        (22,0): 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806, 32.0713,
        (23,0): 107.807, 71.8268, 42.9344, 22.1535, 8.10802, 88.9878, 56.0806
        
