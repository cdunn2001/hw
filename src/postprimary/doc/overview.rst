=============================
PPA - Design and Architecture
=============================

PacBio nomenclature
===================

PacBio SMRT sequencing operates within a silicon chip (a **SMRTcell**)
fabricated to contain a large number of microscopic holes (**ZMWs**,
or **zero-mode waveguides**), each assigned a **hole number**.

Within a ZMW, PacBio SMRT sequencing is performed on a circularized
molecule called a **SMRTbell**. The SMRTbell, depicted below, consists
of:

- the customer's double-stranded DNA **insert** (with sequence
  :math:`I`, read following the arrow)
- (optional) double-stranded DNA **barcodes** (sequences :math:`B_L,
  B_R`) used for multiplexing DNA samples.  While the barcodes are
  optional, they must be present at both ends if present at all.
  Barcodes may or may not be *symmetric*, where symmetric means
  :math:`B_L = B_R^{RC}`.
- SMRTbell **adapters** (sequences :math:`A_L, A_R`), each consisting
  of a double stranded stem and a single-stranded hairpin loop.
  Adapters may or may not be *symmetric*, where symmetric means
  :math:`A_L = A_R`.


.. figure:: img/smrtbell.*
   :width: 100%

   A schematic drawing of a SMRTbell

SMRT sequencing interrogates the incorporated bases in the product
strand of a replication reaction.  Assuming the sequencing of the
template above began at START, the following sequence of bases would
be incorporated (where we are using the superscripts C, R, and RC to
denote sequence complementation, reversal, and
reverse-complementation):

.. math::

   A_L^C B_L^C I^C B_R^C A_R^C B_R^R I^R B_L^R A_L^C \ldots

(note the identity :math:`(x^{RC})^C = x^R`).

The **ZMW read** or **unrolled read** is the output of the
instrument/basecaller upon observing this series of incorporations,
subject to errors due to optical and other limitations.  **Adapter
regions** and **barcode regions** are the spans of the ZMW read
corresponding to the adapter and barcode DNA.  The **subreads** are
the spans of the ZMW read corresponding to the DNA insert.

One complication arises when one considers the possibility that a ZMW
might not contain a single sequencing reaction.  Indeed it could could
contain zero---in which case the ensuing basecalls are a product of
background noise---or it could contain more than one, in which case
the basecall sequence represents two intercalated reads, effectively
appearing as noise.  To remove such noisy sequence, the **high quality
(HQ) region finder** in PostPrimary algorithmically detects a maximal
interval of the ZMW read where it appears that a single
sequencing reaction is taking place.  This region is designated the
**HQ region**, and in the standard mode of operation, PostPrimary will
only output the subreads detected within the HQ region.

.. figure:: img/zmwread.*
   :width: 100%

   A schematic of the regions designated within a ZMW read

.. note::
   Our coordinate system begins at the first basecall in the
   ZMW read (deemed base 0)---i.e., it is *not* relative to the
   HQ region.  Intervals in PacBio reads are given in end-exclusive
   ("half-open") coordinates.  This style of coordinate system should
   be familiar to Python or C++ STL programmers.


BAM everywhere
==============

*Unaligned* BAM files representing the *subreads* will be produced
natively by the PacBio instrument.  The subreads BAM will be the
starting point for secondary analysis.  In addition, the *scraps*
arising from cutting out adapter and barcode sequences will be
retained in a ``scraps.bam`` file, to enable reconstruction of HQ
regions of the ZMW reads, in case the customer needs to rerun
barcode finding with a different option.

The circular consensus tool/workflow (CCS) will take as input an
unaligned subreads BAM file and produce an output BAM file containing
unaligned *consensus* reads.

Alignment (mapping) programs take these unaligned BAM files as input
and will produce *aligned* BAM files, faithfully retaining all tags
and headers.



The role of PostPrimary
=======================

.. figure:: img/postprimaryOverview.*
   :width: 100%

Post-primary is a suite of tools that operates between *primary
analysis* (the basecaller) and *secondary analysis* (genome assembly,
resequencing, and other bioinformatics tools).

The main role of postprimary is to load the ZMW reads from the
basecaller output, extract and label the subreads (by algorithmically
detecting the adapters and barcodes), and output a spec-compliant
subreads BAM file ready for secondary analysis.  These steps cannot be
performed by the basecaller itself due to its streaming architecure,
whereby it only sees one time window at once.  The postprimary tools
can be executed on or off the instrument. For production, for sequel 
postprimary will be performed on the instrument and for RS off the
instrument.

Post-primary algorithms will be implemented generically so they can
operate on a ZMW read in memory, whether it originates from a
BAZ, BAX, or BAM file (the exception is the HQ region finder, which
requires additional information present only in the BAZ file). This
allows to reconstruct ZMW reads in memory and then recall 
barcodes or find controls. For this, postprimary provides user-friendly
command-line interfaces.

.. figure:: img/bam2bam.*
   :width: 100%

The post-primary toolchain will not be able to paper over any
calibration issues that may arise in primary analysis, because only
primary analysis has a full set of metrics at its disposal.  The input
to post-primary must be well calibrated to ensure proper operation of
post-primary algorithms.

The tools are as follows:

``baz2bam``
    converts a BAZ file to BAM file(s), either (subreads + scraps) files
    or HQ region, selectable on command line.  HQ region finding is
    required in either case, but in the former case adapter and
    barcode finding/labeling is also required.  This tool will operate
    only on the instrument.

``bax2bam``
   converts a legacy ``bax.h5`` file to a BAM file, either containing
   a record per-HQ region or per-subread.

``bam2bam``
   postprocesses a BAM file in one convention, producing a BAM file in
   chosen convention.  One motivation for the existence of this as a
   separate tool is the situation where a customer forgets to tell the
   instrument to look for barcodes, or tells it the wrong barcodes,
   and we get a BAM file that is wrong, so we need to reprocess the
   subreads + scraps.

   - (subreads + scraps) -> full HQ region
   - full HQ region      -> subreads + scraps
   - (subreads + scraps) -> (subreads + scraps) [ useful for rerunning
     the adapter/barcoding/controls, without intermediate file ]