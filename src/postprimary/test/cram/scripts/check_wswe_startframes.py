#! /usr/bin/env python

import sys
from pbcore.io import BamReader, IndexedBamReader

def check_records(fname):
    # Open readers
    fh = BamReader(fname)

    for rec in fh:
        pc = rec.peer.opt('pc')
        sf = rec.peer.opt('sf')
        ws = rec.peer.opt('ws')
        we = rec.peer.opt('we')
        firstBaseIndex = -1
        lastBaseIndex = -1
        for i, pulse in enumerate(pc):
            if pulse in 'ACGT':
                if firstBaseIndex == -1:
                    firstBaseIndex = i
                lastBaseIndex = i
        if ws != sf[firstBaseIndex]:
            print "hole: {}".format(rec.peer.opt('zm'))
            print pc
            print sf
            print "ws:        {} we: {}".format(ws, we)
            print "should be: {}   : {}".format(sf[firstBaseIndex],
                                                sf[lastBaseIndex])
            raise ValueError("Wall Start tag does not match startframe values")
        if we != sf[lastBaseIndex]:
            print "hole: {}".format(rec.peer.opt('zm'))
            print pc
            print sf
            print "ws:        {} we: {}".format(ws, we)
            print "should be: {}   : {}".format(sf[firstBaseIndex],
                                                sf[lastBaseIndex])
            raise ValueError("Wall End tag does not match startframe values")

def compare_records(testfname, reffname):
    # Open readers
    tfh = IndexedBamReader(testfname)
    rfh = IndexedBamReader(reffname)
    numSeen = 0

    for (trec, rrec) in zip(tfh, rfh):
        assert trec.holeNumber == rrec.holeNumber
        refws = rrec.peer.opt('ws')
        refwe = rrec.peer.opt('we')
        testws = trec.peer.opt('ws')
        testwe = trec.peer.opt('we')
        if abs(refws - testws) > 4096:
            print "Reference: {}".format(refws)
            print "Test     : {}".format(testws)
            print "Diff     : {}".format(abs(refws - testws))
            raise ValueError("Approximate Wall Start tag does not match "
                             "actual Wall Start to within 4096 frames")
        if abs(refwe - testwe) > 4096:
            print "Reference: {}".format(refwe)
            print "Test     : {}".format(testwe)
            print "Diff     : {}".format(abs(refwe - testwe))
            raise ValueError("Approximate Wall End tag does not match "
                             "actual Wall End to within 4096 frames")
        numSeen += 1
    assert numSeen > 0

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fname = sys.argv[1]
        check_records(fname)
    elif len(sys.argv) == 3:
        testfname = sys.argv[1]
        reffname = sys.argv[2]
        compare_records(testfname, reffname)
    else:
        print ("Please specify a single internal bam, or one internal and one "
               "production bam for comparison")
        exit(1)
