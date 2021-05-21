Algorithm - Control Filtering
-----------------------------

The Control-Filter tool is a customized Sparse Dynamtic
Programming (SDP) aligner, which draws the bulk of it's
core logic from Lance Hepler's re-write of Patrick Mark's
C# SDP implementation in PacBio.Align.  

In the current production implementation, the ControlFilter
class is instantiated with a list of Fasta records.  Once
instantiated, the interface functions of the ControlFilter
object allow users to query whether a given sequence posesses
any significant similarities to any of those initial records.
The result is reported as a simple True / False.

To reduce unnecessary computation:

1) Exit as soon as a single significant hit is detected - a single hit
   is sufficient for us to discard a ZMW as a Control, so we can save
   time by ignoring all further hits.

2) Consider a hit as automatically significant if it's seed chain
   exceeds some pre-defined length and density.  By default at other
   points in the process we discard sequences below an certain ReadScore,
   which should roughly correlate with expected accuracy, giving us an
   an expected lower-bound.  So any seed chain we predict will have a 
   alignment accuracy significantly above that can be assumed positive,
   and we can skip the computationally intensive process of performing
   the full banded S-W alignment.

These were two of the the driving concerns behind not using Blasr, which 
in addition to the complexity of the code base, can not be made to take 
either of these short-cuts, thus significantly increasing the minimal
compute time for a ZMW.