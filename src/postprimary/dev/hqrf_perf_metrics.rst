HQRF Performance Metrics
------------------------

The metrics procedure is as follows

1. generate a BAM file, which will create subreads.bam, each of which will contain a HQR
2. generate new "noHQ" BAM file, using bam2bam to stitch everything back together
3. Align the "noHQ" BAM against a library and determine the "reference" HQ regions
4. compare the HQRs found in #1 with the HQRs found in #3.
   a. An HQR data set consists of regions identified as A2, A1 or A0, which indicates polymerase activity and/or loading.

        A2 = overloaded, A1 = just right loading and sequencing, A0 = no sequencing (no load or dead polymerase)

   b. On a frame by frame base, the number of possible combinations between the two HRQS is a 3x3 = 9 possible states

        A22, A21, A20,
        A12, A11, A10,
        A02, A01, A00

   c. A metric can be formed by counting the number of frames for each of the states, Axy, and filling in the matrix
   d. For ideal loading and sequencing conditions, A22 and A00 are zero.
   e. For ideal HQR finding, all the off diagonals are zero.

Possible derived metrics
========================

All these metrics should be close to zero. They are all normalized to the number of frames, so assuming chemistry is the
same, and the reactions are the same, this allows for cross comparison between ZMWs and between chips.

      Overall agreement                   : (A21+A20+A10+A12+A02+A01)/(frames)
      Systematic bias                     : (A21+A20+A10 - A12 - A02- A01)/frames
      False HQ (called as HQR but was not): (A21 + A01)/frames
      Lost HQ: (called as junk but was HQ): (A12 + A10)/frames
      Sanity                              : (A02 + A20)/frames

      Final metric: TBD. could be weighted sum of above values, or weighted RMS of above values








