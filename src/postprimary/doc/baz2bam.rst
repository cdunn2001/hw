Tool - baz2bam
==============

This part describes the internals of the BAZ to BAM conversion.

In its standard mode of operation on the instrument **baz2bam**
generates an output BAM file containing unaligned subreads culled to
the HQ region, and another BAM containing "scraps", which are the
extents of the ZMW read that do not make it into subreads
(sequences matching the control, sequences outside the HQ region,
adapter sequences).

Through the use of the available flags, the user can instead specify
output of full ZMW reads or HQ regions.  This can be useful for
debugging purposes, where the user can invoke **bam2bam** subsequently
to perform adapter finding and control filtering analyses.


Command line interface
^^^^^^^^^^^^^^^^^^^^^^

Usage::

  Usage: -o outputPrefix -m rmd.xml [options] input.baz

  baz2bam converts the intermediate BAZ file format to BAM, FASTA, AND FASTQ.

  Options:
    -h, --help            show this help message and exit
    -v, --version         Print the tool version and exit
    --progress            Post progress messages via ipc

    Mandatory parameters:
      -o STRING           Prefix of output filenames
      -m STRING, --metadata=STRING
                          Runtime meta data filename

    Optional parameters:
      -j INT              Number of threads for parallel ZMW processing
      -b INT              Number of threads for parallel BAM compression
      --silent            No progress output.
      -Q STRING           Fake HQ filename [simulation file only]

    HQRegion only mode --
    specifying this flag disables polymerase output and activates hqregion output.:
      --hqregion          Output *.hqregions.bam and *.lqregions.bam.

    Adapter finding parameters --
    specifying this flag enables adapter finding and subread output:
      --adapters=adapterSequences.fasta
      --minSoftAccuracy=FLOAT
      --minHardAccuracy=FLOAT

    Control sequence filtering parameter --
    specifying this flag enables control sequence filtering:
      --controls=controlSequences.fasta
      --maxControls=INT   Number of subreads that are used to determine if ZMW is
                          a control. Default: 3

    Additional output read types:
      --fasta             Output fasta.gz
      --fastq             Output fastq.gz

    Fine tuning:
      --minAdapterScore=INT
                          Minimal score for an adapter. Default: 20
      --minPolyLength=INT
                          Minimal polymeraseread length. Default: 1
      --minSubLength=INT  Minimal subread length. Default: 1
      --noStats           Don't compute stats
      --noBam             Don't store bam files
      --noHQ              Disable HQRF; entire polymerase read will be deemed 'HQ'
      --minSnr=FLOAT      Minimal SNR across channels. Default: 4

    White list:
      --whitelistZmwNum=RANGES
                          Only process given ZMW NUMBERs
      --whitelistZmwId=RANGES
                          Only process given ZMW IDs

Workflow
--------

The internal workflow as a schematic:

.. image:: img/baz2bam.*
   :width: 100%

**baz2bam** operates in two stages. In the first stage, headers of all
ZMW_SLICES are read and stored in memory.  In the second stage

 * one threads reads + stitches a batch of 1024 ZMWs and stores them in 
   a StitchedZmw buffer
 * at least one thread takes 100 ZMWs from the buffer, processes them,
   and stores them in the ResultPacket buffer
 * one thread saves BamRecords from the ResultPacket buffer to
   subreads.BAM and scraps.BAM, subreads.FASTA and subreads.FASTQ, and stats.xml.

