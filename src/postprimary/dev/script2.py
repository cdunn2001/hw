#!/usr/bin/env python
import sys
from pbcore.io import *
import collections
from tqdm import *


fields= [ "movieName", "holeNumber", "qStart", "qEnd" , "aStart", "aEnd", "referenceStart" , "referenceEnd", "identity",
          "hqStart", "hqEnd","subreads" ,"hqRegionSnr0","hqRegionSnr1","hqRegionSnr2","hqRegionSnr3"]
print ",".join(map(str,fields))

def makehash():
      return collections.defaultdict(makehash)

db = makehash()

with IndexedBamReader(sys.argv[1]) as f:
    if f.isEmpty:
      raise RuntimeError("empty file" + sys.argv[1])

    for r in tqdm(f) :
        db [ r.holeNumber ]["hqStart"] = r.qStart;
        db [ r.holeNumber ]["hqEnd"] = r.qEnd;
        db [ r.holeNumber ]["hqRegionSnr0"] = r.hqRegionSnr[0];
        db [ r.holeNumber ]["hqRegionSnr1"] = r.hqRegionSnr[1];
        db [ r.holeNumber ]["hqRegionSnr2"] = r.hqRegionSnr[2];
        db [ r.holeNumber ]["hqRegionSnr3"] = r.hqRegionSnr[3];

movieName = ""

with IndexedBamReader(sys.argv[2]) as f:
    if f.isEmpty:
      raise RuntimeError("empty file" + sys.argv[2])

    movieName = f[0].movieName
    for r in tqdm(f) :
        x = db [ r.holeNumber ]
        if not "qMin" in x:
          x["qMin"] = r.qStart
        else :
          x["qMin"] = min(r.qStart,x["qMin"])
        if not "qMax" in x:
          x["qMax"] = r.qEnd
        else :
          x["qMax"] = max(r.qEnd,x["qMax"])

        if not "aMin" in x:
          x["aMin"] = r.aStart
          x["subreads"] = 1
        else :
          x["aMin"] = min(r.aStart,x["aMin"])
          x["subreads"] = x["subreads"] + 1
        if not "aMax" in x:
          x["aMax"] = r.aEnd
        else :
          x["aMax"] = max(r.aEnd,x["aMax"])
        x["referenceStart"] = r.referenceStart;
        x["referenceEnd"] = r.referenceEnd;
        x["identity"] = r.identity

# polymerase read
with IndexedBamReader(sys.argv[3]) as f:
    if f.isEmpty:
      raise RuntimeError("empty file" + sys.argv[3])

    for r in tqdm(f) :
        x = db [ r.holeNumber ]
        x["hqRegionSnr0"] = r.hqRegionSnr[0];
        x["hqRegionSnr1"] = r.hqRegionSnr[1];
        x["hqRegionSnr2"] = r.hqRegionSnr[2];
        x["hqRegionSnr3"] = r.hqRegionSnr[3];

for k in sorted(db):
        v = db[k]
        for z in ("qMin","qMax","aMin","aMax","referenceStart","referenceEnd","identity","hqStart","hqEnd","hqRegionSnr0","hqRegionSnr1","hqRegionSnr2","hqRegionSnr3") :
          if not z in v:
            v[z] = "NA"
        if not "subreads" in v:
          v["subreads"] = 0
        fields= [ movieName, k, v["qMin"], v["qMax"], v["aMin"], v["aMax"], v["referenceStart"], v["referenceEnd"], v["identity"], v["hqStart"], v["hqEnd"] ,v["subreads"], v["hqRegionSnr0"], v["hqRegionSnr1"],v["hqRegionSnr2"],v["hqRegionSnr3"] ]
        print ",".join(map(str,fields))
    
