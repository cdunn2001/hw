#! /usr/bin/env python

import sys
from pbcore.io import BamReader


def compare_records(qRec, tRec, i):
    # Compare required / columnar elements
    if tRec.qName      != qRec.qName:
        raise ValueError("Mismatch at Record #{0} - QNAMEs don't match ({1} and {2})".format(i, tRec.qName, qRec.qName))
    if tRec.peer.flag  != qRec.peer.flag:
        raise ValueError("Mismatch at Record #{0} ({1}) - FLAGs don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.flag, qRec.peer.flag))
    if tRec.peer.rname != qRec.peer.rname:
        raise ValueError("Mismatch at Record #{0} ({1}) - RNAMEs don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.rname, qRec.peer.rname))
    if tRec.peer.pos   != qRec.peer.pos:
        raise ValueError("Mismatch at Record #{0} ({1}) - POSes don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.pos, qRec.peer.pos))
    if tRec.peer.mapq  != qRec.peer.mapq:
        raise ValueError("Mismatch at Record #{0} ({1}) - MAPQs don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.mapq, qRec.peer.mapq))
    if tRec.peer.cigar != qRec.peer.cigar:
        raise ValueError("Mismatch at Record #{0} ({1}) - CIGARs don't match".format(i, tRec.qName))
    if tRec.peer.rnext != qRec.peer.rnext:
        raise ValueError("Mismatch at Record #{0} ({1}) - RNEXTs don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.rnext, qRec.peer.rnext))
    if tRec.peer.pnext != qRec.peer.pnext:
        raise ValueError("Mismatch at Record #{0} ({1}) - PNEXTs don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.pnext, qRec.peer.pnext))
    if tRec.peer.tlen  != qRec.peer.tlen:
        raise ValueError("Mismatch at Record #{0} ({1}) - TLENs don't match ({2} and {3})".format(i, tRec.qName, tRec.peer.tlen, qRec.peer.tlen))
    if tRec.peer.seq   != qRec.peer.seq:
        raise ValueError("Mismatch at Record #{0} ({1}) - SEQs don't match".format(i, tRec.qName))
    if tRec.peer.qual  != qRec.peer.qual:
        raise ValueError("Mismatch at Record #{0} ({1}) - QUALs don't match".format(i, tRec.qName))

    # Compare optional / tag elements
    tags = dict(qRec.peer.tags)
    if len(tags) < len(qRec.peer.tags):
        raise ValueError("Mismatch at Record #{0} ({1}) - Duplicate tags detected".format(i, tRec.qName))

    targetTagCount = 0
    for tag, value in tRec.peer.tags:
        # Catch missing tags
        try:
            rValue = tags[tag]
        except KeyError:
            # the ws, we tags are removed whenever region boundaries change. We
            # accept their absence as a result of running bam2bam, even though
            # we're not checking whether they should be missing (in this test)
            if tag not in ['ws', 'we']:
                raise ValueError("Mismatch at Record #{0} ({1}) - Tag '{2}' missing".format(i, tRec.qName, tag))
            else:
                continue
        targetTagCount += 1

        # Catch known floating-point tags and round before testing
        if isinstance(value, float):
            if (round(value, 6) != round(rValue, 6)):
                raise ValueError("Mismatch at Record #{0} ({1}) - Values for Tag '{2}' don't match".format(i, tRec.qName, tag))
        # Otherwise just compare the tags as-is
        elif value != rValue:
            raise ValueError("Mismatch at Record #{0} ({1}) - Values for Tag '{2}' don't match".format(i, tRec.qName, tag))

    if len(tags) != targetTagCount:
        raise ValueError("Mismatch at Record #{0} ({1}) - Extra tags detected".format(i, tRec.qName))


def compare_bam_files(query_file, target_file):
    # Open readers
    tReader = BamReader(target_file)
    qReader = BamReader(query_file)

    # Iterate and compare
    idx = 0
    for qRec, tRec in zip(iter(qReader), iter(tReader)):
        idx += 1
        compare_records(qRec, tRec, idx)


target_file = sys.argv[1]
query_file  = sys.argv[2]
compare_bam_files( query_file, target_file )
