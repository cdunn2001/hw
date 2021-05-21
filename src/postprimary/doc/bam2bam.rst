Tool - bam2bam
==============

The **bam2bam** tool operates on BAM files in one convention, allowing
reprocessing and optional conversion to another convention.  For
example, a BAM containing HQ regions could be processed, adapter hits
identified, and a new subreads BAM file produced.

The necessity for this tool arises from the scenario where the
customer has initially caused an incorrect invocation of PostPrimary
on the instrument, for example forgetting to request control analysis,
and therefore needs to reanalyze the data with control finding enabled.

This tool presents a grid of *input convention* / *output convention*,
and we likely will not want or need to support or even implement every
entry in this grid.

Both production and pulse BAM files can be processed.

Availability
^^^^^^^^^^^^

bam2bam is installed on every Sequel machine, gets shipped with smrtanalysis,
and can be installed via brew::
  
  brew install PacificBiosciences/tools/bam2bam

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

Usage::

  Usage: -o outputPrefix [options] input.(subreads|hqregion).bam input.(scraps).bam

  bam2bam operates on BAM files in one convention (subreads+scraps or
  hqregions+scraps), allows reprocessing (for example, with a different set of
  adapter sequences than was originally used) and then outputs the resulting BAM
  files in the desired convention (subreads/hqregions/zmw plus scraps).

  "Scraps" BAM files are always required to reconstitute the zmw reads
  internally. Conversely, "scraps" BAM files will be output.

  ZMW reads are not allowed as input, due to the missing HQ-region annotations!

  Input read convention is determined from the READTYPE annotation in the @RG::DS
  tags of the input BAM files.

  Options:
    -h, --help            show this help message and exit
    -v, --version         Print the tool version and exit

    Mandatory parameters:
      -o STRING           Prefix of output filenames

    Optional parameters:
      -j INT              Number of threads for parallel ZMW processing
      -b INT              Number of threads for parallel BAM compression
      --silent            No progress output.

    BAM conventions:
      --zmw        Create a zmw reads.
      --hqregion          Output *.hqregions.bam and *.scraps.bam.

    Adapter finding parameters:
      --adapters=adapterSequences.fasta
      --minSoftAccuracy=FLOAT
      --minHardAccuracy=FLOAT

    Parameters needed for control sequence filtering --
    specifying this flag enables control sequence filtering:
      --controls=controlSequences.fasta
      --maxControls=INT   Number of subreads that are used to determine if ZMW is
                          a control. Default: 3

    Additional output read types:
      --fasta             Output fasta.gz
      --fastq             Output fastq.gz
      --noBam             Do NOT produce BAM outputs.

    Fine tuning:
      --minAdapterScore=int
                          Minimal score for an adapter
      --minPolyLength=INT
                          Minimal zmwread length. Default: 1
      --minSubLength=INT  Minimal subread length. Default: 1
      --fullHQ            Disable HQRF; entire zmw read will be deemed 'HQ'

  Example: bam2bam in.subreads.bam in.scraps.bam -o out --adapters adapters.fasta


Example command-line invocations:

- A new adapter finding algorithm can be performed on an older data set::

    $ bam2bam --adapter adapters.fasta                  \
              -o movieName.newAdapters                  \
              movieName.subreads.bam movieName.scraps.bam

- For internal analysis, the user wants to have convert subreads+scraps
  to zmw reads::

    $ bam2bam --zmw                                     \
              -o movieName.stitched                     \
              movieName.subreads.bam movieName.scraps.bam

- The user wants to have stitched zmw reads in FASTA.GZ format
  (a \*.zmw.BAM is automatically created)::

    $ bam2bam --zmw                                     \
              --fasta                                   \
              -o movieName.stitched                     \
              movieName.subreads.bam movieName.scraps.bam

- Convert subreads+scraps to hqregions+scraps::

    $ bam2bam --hqregion                                \
              -o movieName.stitchedHQ                   \
              movieName.subreads.bam movieName.scraps.bam

- The user needs the hqregions also in FASTA.GZ format 
  (a \*.hqregions.BAM is automatically created)::

    $ bam2bam --hqregion                                \
              --fasta                                   \
              -o movieName.stitched                     \
              movieName.subreads.bam movieName.scraps.bam

- The user needs subreads in FASTA.GZ format only::

    $ bam2bam --nobam                                   \
              --fasta                                   \
              -o movieName.new                          \
              movieName.subreads.bam movieName.scraps.bam

- Convert hqregions+scraps to subreads+scraps with adapter::

    $ bam2bam --adapter adapters.fasta                   \
              -o movieName.newVersion                    \
              movieName.hqregions.bam movieName.scraps.bam

- Sanity check that the output is the same as the input, plus a new
  BAM header entry with the bam2bam version::

    $ bam2bam -o movieName.sanity                       \
              movieName.subreads.bam movieName.scraps.bam

    $ samtools view movieName.subreads.bam > a1
    $ samtools view movieName.sanity.subreads.bam > b1
    $ diff a1 b1

    $ samtools view movieName.scraps.bam > a1
    $ samtools view movieName.sanity.scraps.bam > b1
    $ diff a1 b1

    $ rm a1 b1

- User forgot to tell instrument to filter controls and
  wants perform spike-in control filtering on his computer::

    $ bam2bam --controls control_orig.fasta                   \
              -o movieName.control_orig                    \
              movieName.subreads.bam movieName.scraps.bam         

- User found a better reference for his spike-in controls::

    $ bam2bam --controls control_better.fasta                    \
              -o movieName.control_better                     \
              movieName.subreads.bam movieName.scraps.bam

- Perform complete analysis from scratch, as PPA was released with a 
  new set of improved algorithms. The only thing that cannot be computed
  from scratch are HQ-regions::

    $ bam2bam --adapter adapters.fasta                    \
              --controls control.fasta                    \
              -o movieName.newPPAVersion                  \
              movieName.subreads.bam movieName.scraps.bam

- Treat the complete ZMW read as a HQ region and perform
  adapter finding::

    $ bam2bam --fullHQ                                    \
              --adapter adapters.fasta                    \
              -o movieName.fullhq                         \
              movieName.subreads.bam movieName.scraps.bam
