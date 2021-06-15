#! /usr/bin/env python

import sys

def read_csv( filename ):
    data = {}
    with open( filename ) as handle:
        handle.next()
        for line in handle:
            parts = line.strip().split(',')
            read = parts[0]
            adp = (int(parts[1]), int(parts[2]))
            try:
                data[read].append( adp )
            except:
                data[read] = [adp]
    return data

def zmw_counts( file1, file2 ):
    diffs = {}

    d1 = read_csv( file1 )
    d2 = read_csv( file2 )

    for key, adps1 in d1.iteritems():
        adps2 = d2[key]

        diff = len(adps1) - len(adps2)
        try:
            diffs[diff].append( key )
        except:
            diffs[diff] = [key]

    return diffs

def zmw_diffs( file1, file2 ):
    diffs = {}

    d1 = read_csv( file1 )
    d2 = read_csv( file2 )

    total   = 0
    matches = 0
    overlap = 0
    missing = 0
    extra   = 0

    for key, adps1 in d1.iteritems():
        adps2 = d2[key]
        adp2_idx = 0

        if len(adps2) > len(adps1):
            extra += len(adps2) - len(adps1)

        for adp1 in adps1:
            total += 1
            old_idx = adp2_idx
            for adp2 in adps2[adp2_idx:]:
                if adp1[0] == adp2[0] and adp1[1] == adp2[1]:
                    matches += 1
                    adp2_idx += 1
                    break
                if adp1[0] < adp2[1] and adp1[1] > adp2[0]:
                    overlap += 1
                    adp2_idx += 1
                    break
                if adp1[1] < adp2[0]:
                    break
            if adp2_idx == old_idx:
                missing += 1

    return {'perfect': matches/float(total),
            'match': (matches + overlap)/float(total),
            'overlap': overlap/float(total),
            'missing': missing/float(total),
            'extra': extra/float(total)}

res = zmw_diffs(sys.argv[1], sys.argv[2])

for key in sorted(res.keys()):
    print key, res[key]
