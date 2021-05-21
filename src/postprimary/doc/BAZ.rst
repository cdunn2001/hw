BAZ - Format Specification
##########################

Primary scope of this document is the specification of the intermediate base 
file employed in the post-primary analysis (PPA) step of the Sequel workflow. 
Secondary scope is the API and demo code of writing a BAZ file, performed by 
**BazWriter** in the primary analysis (PA).
Tertiary scope is the API of reading a BAZ file, performed by :ref:`baz2bam` 
in PPA.
Algorithms for processing ZMW reads to customer-friendly output is 
described under :ref:`polylabel`.
Calculations of the per ZMW and per chip statistics is described under 
:ref:`bazsts`.

The previous generation RSII produces a single movie that gets processed by PA.
For a given ZMW, all base calls are available at once for processing. 
In this processing step, the big four are employed

* AdapterFinder,
* BarcodeCaller,
* ControlFilter,
* HQ-RegionFinder.

These algorithms require knowledge about the full-lenth ZMW 
read, i.e., the read of a single ZMW from a full-time movie.
The throughput of Sequel will be up to eight times higher, which causes new 
challenges for the data processing and storage. 
In addition, to minimize the time from starting a sequencing run to providing 
the final customer data output, primary data aquisition is performed in a 
streaming workflow using FPGAs and PhiXs. 
**Primary provides readouts of all ZMWs of a 16384 frames (163.84 seconds)
window, 32 chunks a 512 frames, called a super-chunk**.

.. figure:: img/primary_data.*
   :width: 100%

The optical data is processed by PA in real-time, but the big four cannot be
performed in real-time nor on individual super-chunks,
as the full ZMW read is needed. 
One possible solution is storing everything in RAM, but this solution is very
expensive as we would need ~250 GB for a single run and consequently >500gb to
allow continuous sequencing.
Instead, each super-chunk will to be written to disk by PA in a binary format 
called **BAZ**. A super-chunk contains the individual ZMW-slices. 
After a sequencing run has finished, this file will be read once by PPA to:

* stitch individual ZMWs from multiple slices,
* find the maximal HQ-regions, 
* filter spike-in controls, 
* call barcodes,
* find adapters, 
* store  records as BAM, and
* compute and store per ZMW and per chip statistics. 


BAZ file (intermediate basecall file) format specification
==========================================================

This part describes the challenges in storing the intermediate base calls,
the design of the file format, the C++ API, implementation hints, 
and corresponding benchmarks. 
The resulting library is called **BazIO**.

Visibilty
---------

The BAZ file is an intermediate file, not customer-visible, 
and serves as an advanced swap file.

Field usage
^^^^^^^^^^^
The BAZ file will not be transferred off the machine and will immediately be 
deleted after conversion by PPA and after the BAM files have been checked
for validity.

Internal usage
^^^^^^^^^^^^^^
The BAZ file might be transferred off the instrument to allow algorithm
development that depends on block metrics, e.g., HQ-RegionFinder.

Definitions
-----------

*ZMW-slice*
  all information about a ZMW within a 16384 frames slice

*super-chunk*
  information about 1M ZMW-slices

*readout* / *event*
  a base or pulse call (depends on the mode)

BAZ file format
---------------

The BAZ format is in super-chunk-read-major.
Each super-chunk contains up to 1,171,456 ZMW-slices of a fully loaded chip,
including pseudo ZMWs in the border region.
The upper limit for the number of ZMWs is a 32-bit unsigned integer.
Each ZMW-slice can contain base and pulse calls, associated QVs, tags, 
and kinetics. In addition lower resolution metrics can be stored in three 
different frequencies, e.g., to determine HQ-regions or average PKMIDs.

The BAZ format is designed to provide maximal flexibility. 
The movie length is variable and does not need to be set prior acquisition,
as each super-chunk gets appended to the file.
For each super-chunk, the number of ZMW-slices is variable.
For each ZMW-slice, the number of base and/or pulse calls and associated number 
of metric blocks are variable.
The information that are saved per readout and for each metric-frequency
can be set per acquisition. This allows running different sequencing protocols
with the same workflow.

File format specification, each row represents one super-chunk:

.. figure:: img/baz_fileformat.*
   :width: 100%

In each super-chunk, every 1024th ZMW-slice is 4k aligned.
To retrieve one ZMW read, all ZMW-slices of this particular ZMW are 
stitched; in a three hour movie, 66 slices have to be read, i.e., 66 disk seeks.
To minimize the number of seeks, ZMW-slices are read in batches of 1024.

The BAZ file contains SANITY DWORDs to guarantee that the file is not corrupt
and serve as orientation points in the file.

Each super-chunk starts with a SUPER_CHUNK_META that stores the offset to the 
next super-chunk and the number of ZMW-slices.
For each ZMW-slice, a ZMW_SLICE_HEADER stores the offset to the actual data,
the hole number, size of the packet byte stream, number of bases/pulses, 
and number of each frequency metric block.
Each ZMW-slice contains a packet byte stream of base calls or pulses 
and metrics in three different frequencies.

File header
-----------

The file header consists of a large block of ASCII, formatted as JSON.
To conform with PacBio JSON practices, the header is wrapped in a meta header,
that has required TYPE and HEADER fields.
The HEADER block describes:

* BAZ version major.minor.patch
* BaseCaller and BazWriter version
* P4 CL
* Movie name
* The layout and field bit sizes of the per readout packet
* Three metric blocks, each with the layout of fields and their bit sizes and 
  the metric frequency
* ZMW-slice length in frames
* Super-chunk length in frames
* Frame rate in Hz as a float
* File footer offset
* ZMW id to ZMW number mapping
* List of sentinel ZMW numbers
* List of per ZMW unit features

The file header is followed by a SANITY DWORD. The first SUPER_META_CHUNK is 4k aligned.
At least the readout has to be specified per packet.
All three metric blocks are optional.

.. highlight:: javascript

Example::

    {
        "HEADER" : {
            "BASE_CALLER_VERSION" : "3.0.17.173393",
            "BAZWRITER_VERSION" : "2.5.1.173160",
            "BAZ_MAJOR_VERSION" : 1,
            "BAZ_MINOR_VERSION" : 3,
            "BAZ_PATCH_VERSION" : 0,
            "COMPLETE" : 1,
            "FILE_FOOTER_OFFSET" : 19676020,
            "FRAME_RATE_HZ" : 100,
            "HF_METRIC" : {
                "FIELDS" : [
                        [ "NUM_BASES", 16, false, 1 ],
                        [ "NUM_FRAMES", 16, false, 1 ]
                ],
                "FRAMES" : 1024
            },
            "LF_METRIC" : {
                "FIELDS" : [
                    [ "NUM_BASES", 16, false, 1 ],
                    [ "NUM_FRAMES", 16, false, 1 ],
                    [ "PKMID_A", 16, true, 10 ],
                    [ "PKMID_C", 16, true, 10 ],
                    [ "PKMID_G", 16, true, 10 ],
                    [ "PKMID_T", 16, true, 10 ],
                    [ "PKMID_FRAMES_A", 16, false, 1 ],
                    [ "PKMID_FRAMES_C", 16, false, 1 ],
                    [ "PKMID_FRAMES_G", 16, false, 1 ],
                    [ "PKMID_FRAMES_T", 16, false, 1 ],
                    [ "BASELINE_RED_SD", 16, true, 10 ],
                    [ "BASELINE_GREEN_SD", 16, true, 10 ],
                    [ "BASELINE_RED_MEAN", 16, true, 10 ],
                    [ "BASELINE_GREEN_MEAN", 16, true, 10 ],
                    [ "PULSE_WIDTH", 16, false, 1 ],
                    [ "BASE_WIDTH", 16, false, 1 ],
                    [ "PIXEL_CHECKSUM", 16, true, 1 ]
                ],
                "FRAMES" : 16384
            },
            "MF_METRIC" : {
                "FIELDS" : [
                    [ "NUM_BASES", 16, false, 1 ],
                    [ "NUM_PULSES", 16, false, 1 ],
                    [ "NUM_FRAMES", 16, false, 1 ],
                    [ "NUM_HALF_SANDWICHES", 16, false, 1 ],
                    [ "NUM_PULSE_LABEL_STUTTERS", 16, false, 1 ]
                ],
                "FRAMES" : 4096
            },
            "MOVIE_LENGTH_FRAMES" : 1080000,
            "MOVIE_NAME" : "m00001_052415_013000",
            "P4_VERSION" : "173160",
            "PACKET" : [
                [ "READOUT", 2 ],
                [ "OVERALL_QV", 4 ],
                [ "", 2 ],
                [ "IPD_V1", 8 ],
                [ "PW_V1", 8 ]
            ],
            "SLICE_LENGTH_FRAMES" : 16384,
            "TRUNCATED" : 0,
            "ZMW_NUMBER_LUT" : [
                [ "0x186a0", 1024 ],
                [ "0x30d40", 12 ]
            ],
            "ZMW_UNIT_FEATURES_LUT" : [
                [ 0, 1036 ]
            ]
        },
        "TYPE" : "BAZ"
    }


Production vs. Internal mode
----------------------------

Sequel can be run in four different modes:

- **production** stores base calls and a minimal set of metrics
- **internal** stores pulse calls, an extended set of metrics, and traces

Packets
-------

The FILE_HEAD specifies what fields can be used in a packet.
Following fields can be used in arbitrary order, with the exemption that a 
single field smaller than a byte may not be cross-byte::

   1b  IS_BASE
   1b  IS_PULSE
   2b  READOUT   => {A, C, G, T} = [0, 4)
   3b  DEL_TAG   => {A, C, G, T, N} = [0, 5)
   3b  SUB_TAG
   3b  LABEL
   3b  ALT_LABEL
   4b  DEL_QV    => [0, 16)
   4b  SUB_QV
   4b  INS_QV
   4b  MRG_QV
   4b  LAB_QV
   4b  ALT_QV
   4b  OVERALL_QV
   4b  PULSE_MRG_QV
   8b  IPD_LL    => [0, 255)
   8b  PW_LL
   8b  PKMEAN_LL
   8b  PKMID_LL
   8b  IPD_V1
   8b  PW_V1
   16b IPD16_LL  => [0, 65536)
   16b PW16_LL
   16b PKMEAN16_LL
   16b PKMID16_LL
   32b IPD32_LL
   32b PW32_LL


The size of each field can be adjusted and BazReader will store fields as 32 bit 
signed integers. All fields are unsigned.

Additional fields can be defined and saved, but will be ignored by BazReader in
a similar way as undefined metrics are.

PKMID and PKMEAN are stored after multiplied by ten to keep one decimal place
precision.

Lossless encoding, by variable length encoding, can be activated for 
IPD_LL, PW_LL, PKMEAN_LL, and PKMID_LL  by::

    ["IPD_LL", 8, 255, "IPD16_LL",16]

Field order is: name, bit-size, escape number, name of the next extension, bit-size.

The in-memory representation is defined here::

	//PrimaryAnalysis/main/Sequel/common/pacbio/primary/Basecall.h

	char       base_;
	int8_t     insertionQV_;
	int8_t     deletionQV_;
	char       deletionTag_;
	int8_t     substitutionQV_;
	char       substitutionTag_;
	uint32_t   start_;       // frames
	uint16_t   width_;       // frames
	uint16_t   meanSignal_;  // hundredths of a DN or e-.
	uint16_t   midSignal_;   // hundredths of a DN or e-.
	char       label_;
	int8_t     labelQV_;
	char       altLabel_;
	int8_t     altLabelQV_;

.. highlight:: c++

User-defined metrics
--------------------

Metrics are kinetic properties and averaged over a defined window-size.
Metrics should be mutually exclusively, but don't necessarily have to be, 
assigned to a high-, medium-, or low-frequency metric block.
Currently defined metrics are::

    NUM_FRAMES
    NUM_BASES
    NUM_PULSES
    BASELINE_RED_SD
    BASELINE_GREEN_SD
    BASELINE_RED_MEAN
    BASELINE_GREEN_MEAN
    PULSE_WIDTH
    BASE_WIDTH
    PKMID_A
    PKMID_C
    PKMID_G
    PKMID_T
    NUM_SANDWICHES
    NUM_HALF_SANDWICHES
    NUM_PULSE_LABEL_STUTTERS
    PKMID_FRAMES_A
    PKMID_FRAMES_C
    PKMID_FRAMES_G
    PKMID_FRAMES_T
    PIXEL_CHECKSUM
    GAP

Each metric is reported as::

    [name, bitsize, signed, scalingFactor]

The different frequencies represent the window size in number of frames. 
Medium- and low-frequency size have to be a multiple of high-frequency.
One possible setting could be:

* Low-frequency covers one ZMW-slices, therefore 512*32 frames = 163.84 seconds.
* Medium-frequency will be collected four times per ZMW-slice, 4096 frames = 40.96 seconds.
* High-frequency will be collected 16 times per ZMW-slice, 1024 frames = 10.24 seconds.
  Possibly, high-frequency metrics can be collected every chunk, thus 32 times, 
  512 frames = 5.12 seconds, or even every half chunk.

PA calculates all metrics for high-frequency in the base calling step.
BazWriter consumes the user config, stores it in the FILE_HEAD 
and writes user-defined metrics to disk.
Medium- and low-frequency metrics are averaged on-the-fly, 
weighted over time, #pulses, or #bases.

Additional fields can be added without any need for reimplementation, 
but will ignored by BazReader. Such field will be treated as a GAP field, which
may not be used explicitly. Metrics that are provided by PA, but are not 
assigned in the user config, will be ignored.

Lower frequent metrics frequencies are created by merging multiple MetricBlocks::

	MetricBlock(std::initializer_list<MetricBlock> parents);
	MetricBlock(std::vector<MetricBlock>& parents);
	MetricBlock(MetricBlock* parents, size_t numParents);

The fields are merged differently. For each frequency there is a RATIO_TO_HF::

        Summation:
         - NUM_BASES
         - PKMID_BASES_A
         - PKMID_BASES_C
         - PKMID_BASES_G
         - PKMID_BASES_T
         - NUM_PULSES
         - NUM_SANDWICHES
         - NUM_HALF_SANDWICHES
         - NUM_PULSE_LABEL_STUTTERS

        Average by RATIO_TO_HF:
         - BASELINE_RED_MEAN   
         - BASELINE_GREEN_MEAN   

        Weight each high-frequency field, sum, and average by summed weight:
         - BASE_WIDTH  * NUM_BASES
         - PKMID_A     * PKMID_BASES_A
         - PKMID_C     * PKMID_BASES_C
         - PKMID_G     * PKMID_BASES_G
         - PKMID_T     * PKMID_BASES_T
         - PULSE_WIDTH * NUM_PULSES

        Sum the variance (SD^2), average by RATIO_TO_HF, and sqrt:
         - BASELINE_RED_SD 
         - BASELINE_GREEN_SD    

Due to latency in the real-time streaming, the first and last slice will not 
contain exactly 16384 frames. The first slice will be slightly shorter and the
very last slice makes up for the missing frames of the first slice. All other
slices will contain exactly 16384 frames.

.. image:: img/metricBlocks.*
   :width: 100%

To solve this irregularity, medium- and low-frequency blocks are computed from
high-frequency blocks starting at the beginning the slice. The medium- and 
low-frequency blocks at the end might be longer, w.r.t. the number of 
high-frequency blocks they contain. As a consequence, we cannot guarantee that a 
medium- or low-frequency block is composed of exactly X high-frequency blocks.
Therefore, we store the number of pulses or bases for each single block for each
frequency. This guarantees that we can map from blocks to bases/pulses and
vice versa.

.. _stitchedzmw:

StitchedZmw
-----------

BazReader stitches ZMW-slices and provides one StitchedZmw.h per ZMW.
We can ask a StitchedZmw if it contains a certain PacketField by its
PacketFieldName (postprimary/baz/PacketFieldName.h)::

  bool HasPacketField(PacketFieldName fieldName) const;

We can ask for it's property vector with::

  const std::vector<uint32_t>& PacketField(PacketFieldName fieldName) const;

We define an enum to access Packet fields::

    SMART_BAZ_ENUM(PacketFieldName,
        READOUT       = 0, // Base or pulse readout {A, C, G, T}

        DEL_TAG       = 1, // Deletion Tag       {A, C, G, T, N}
        SUB_TAG          , // Substitution Tag   {A, C, G, T, N}
                           
        LABEL            , // Pulse Label Tag
        ALT_LABEL        , // Pulse Alternative Label Tag

        DEL_QV           , // Deletion QV
        SUB_QV           , // Substitution QV
        INS_QV           , // Insertion QV
        MRG_QV           , // Merge QV

        LAB_QV           , // Label QV
        ALT_QV           , // Alternative Label QV

        IPD_LL           , // Inter Pulse Duration of a base, lossless (8 bit part)
        IPD16_LL         , // Inter Pulse Duration of a base, lossless (16 bit extension)
        IPD32_LL         , // Inter Pulse Duration of a base, lossless (32 bit extension)
        IPD_V1           , // Inter Pulse Duration of a base, lossy

        PW_LL            , // Pulse width of a base, lossless (8 bit part)
        PW16_LL          , // Pulse width of a base, lossless (16 bit extension)
        PW32_LL          , // Pulse width of a base, lossless (32 bit extension)
        PW_V1            , // Pulse width of a base, lossy
        
        PKMEAN_LL        , // PK mean, lossless (8 bit part)
        PKMEAN16_LL      , // PK mean, lossless (16 bit extension)
        
        PKMID_LL         , // PK mid, lossless (8 bit part)
        PKMID16_LL       , // PK mid, lossless (16 bit extension)
        
        IS_BASE          , // Is this event a base?
        IS_PULSE         , // Is this a pulse? Otherwise it was added by P2B

        PX_LL            , // Pulse width of the underlying pulse, lossless (8 bit part)
        PX16_LL          , // Pulse width of the underlying pulse, lossless (16 bit extension)
        PX32_LL          , // Pulse width of the underlying pulse, lossless (32 bit extension)

        PD_LL            , // Pre pulse frames of the underlying pulse, lossless (8 bit part)
        PD16_LL          , // Pre pulse frames of the underlying pulse, lossless (16 bit extension)
        PD32_LL          , // Pre pulse frames of the underlying pulse, lossless (32 bit extension)

        OVERALL_QV       ,

        PULSE_MRG_QV     ,

        START_FRAME      ,

        PKMEAN2_LL       , // PK mean 2, two 16bit channels combined via bitshifting: green << 16 | red

        PKMID2_LL        , // PK mid2, two 16bit channels combined via bitshifting: green << 16 | red

        GAP          = -1
    );

The SMART_BAZ_ENUM allows to map from and to strings.

In the internal mode, not every READOUT is a basecall::

    const std::vector<uint8_t>& BaseCalls() const;

AF, BC, and CF operate only on the base calls. To get back to the pulse
coordinate system to cut subreads and scraps::

    size_t BaseToPulseIndex(size_t index) const;

Same story for high-, medium-, and low-frequency metric fields::

    const std::vector<uint32_t>& HFMField(MetricFieldName fieldName) const;
    bool HasHFMField(MetricFieldName fieldName) const;

    const std::vector<uint32_t>& MFMField(MetricFieldName fieldName) const;
    bool HasMFMField(MetricFieldName fieldName) const;

    const std::vector<uint32_t>& LFMField(MetricFieldName fieldName) const;
    bool HasLFMField(MetricFieldName fieldName) const;
  
    SMART_BAZ_ENUM(MetricFieldName ,
        NUM_FRAMES               = 0,
        NUM_BASES                   ,
        NUM_PULSES                  ,
        BASELINE_RED_SD             ,
        BASELINE_GREEN_SD           ,
        BASELINE_RED_MEAN           ,
        BASELINE_GREEN_MEAN         ,
        PULSE_WIDTH                 ,
        BASE_WIDTH                  ,
        PKMID_A                     ,
        PKMID_C                     ,
        PKMID_G                     ,
        PKMID_T                     ,
        NUM_SANDWICHES              ,
        NUM_HALF_SANDWICHES         ,
        NUM_PULSE_LABEL_STUTTERS    ,
        PKMID_FRAMES_A              ,
        PKMID_FRAMES_C              ,
        PKMID_FRAMES_G              ,
        PKMID_FRAMES_T              ,
        PIXEL_CHECKSUM              ,
        GAP                      = -1
    );

If we are interested in the number of metric blocks for each frequency 
or the number of events::

  uint64_t NumEvents() const;
  uint64_t NumHFMBs() const;
  uint64_t NumMFMBs() const;
  uint64_t NumLFMBs() const;

We can get the metric block number by the event number::

  uint32_t eventToHFMB(uint32_t event) const;
  uint32_t eventToMFMB(uint32_t event) const;
  uint32_t eventToLFMB(uint32_t event) const;

Or vice-versa, the number of the first event in the metric block::

  const std::vector<uint32_t>& HFMBToEvent() const;
  const std::vector<uint32_t>& MFMBToEvent() const;
  const std::vector<uint32_t>& LFMBToEvent() const;

We also provide static maps::

	class MetricFieldMap
	    static std::map<MetricFrequency, std::string> metricFrequencyToString;

	class PacketFieldMap 
	    static std::map<PacketFieldName, std::pair<std::string, FieldType>> packetFieldToBamID;
	    static std::map<PacketFieldName, std::pair<std::string, FieldType>> packetPulseFieldToBamID;

	enum class FieldType
	    CHAR,
	    UINT8,
	    UINT16,
      UINT32


Versioning
----------

BAZ is saved in a *major.minor.patch* fashion.
In case of a BAZ file format change or API breaks, the major version is increased.
The minor version is increased for each official release.
New functionality during the development of a release is indicated by an
increase of the patch number.

Run-time information
--------------------

Run-time, sequencing-run, and machine-specific information will be saved in a 
.run.metadata.xml file by the instrument control system.
The BAZ file only saves information that are important to process its own data.
The user has to provide with the metadata file to baz2bam, to forward defined 
fields into the BAM header.

File size per bit
-----------------

Every bit counts. 
The more compact we can save information, the more time PPA can spend in 
sophisticated algorithms in the big four.
BAZ costs per byte extrapolated to a three hour run on a fully loaded chip 
with 5 bp/s:

- PACKET: ~54GB
- HF_METRIC: ~2GB (5.12s)
- HF_METRIC: ~1GB (10.24s)
- MF_METRIC: ~264MB (40.96s)
- LF_METRIC: ~66MB (163.84s)

Limitations
-----------

The number of base calls per ZMW chunk (163s) is limited to 16 bit, i.e., 
65,535 bases. 
This allows a peak, burst polymerase activity of 400 bp/s that is be beyond
the current and future enzyme activity.



Directory / file layout
-----------------------

.. code-block:: none

    postprimary
    |── include
    │   +── postprimary
    │       |── asterisk2bam
    │       |── bam2bam
    │       |── baz
    │       |── baz2bam
    │       |── bazreader
    │       |── bazsts
    │       |── bazwriter
    │       |── cluster
    │       |── daemon
    │       |── fastaparser
    │       |── frames
    │       |── log
    │       |── pbpartition
    │       |── polylabel
    │       +── simulation
    |── install
    |── template
    |── test
    │   |── cram
    │   │   |── adapter
    │   │   |── data
    │   │   +── scripts
    │   +── unit
    │       +── data
    +── third-party
        |── gtest
        |── htslib
        |── optionparser
        |── pbbam
        |── pbmicrolog
        |── pbsparse
        +── seqan

.. highlight:: c++

For data transfer, we define a lightweight struct to wrap all information
of a single ZmwSlice:

.. literalinclude:: ../../common/pacbio/primary/ZmwSlice.h
    :lines: 47, 85-91

Smart pointers are defined as::

    // Encapsulate manually allocated byte stream memory in a smart pointer
    // that frees on destruction.
    template<typename T> using SmrtMemPtr = std::unique_ptr<T, void (*)(void*)>;

    using SmrtBytePtr = SmrtMemPtr<uint8_t>;

BazWriter.h API
---------------

The preferred API of BazWriter.h::

    /// Creates a new BAZ file and writes file header.
    /// \param filePath          File name of the output BAZ file.
    /// \param fileHeaderBuilder JSON of file header from a builder.
    BazWriter(const std::string& filePath, 
              FileHeaderBuilder& fileHeaderBuilder,
              const size_t maxNumSamples = 100000,
              bool silent = true);

    /// Given primary pulse/base calls and trace metrics
    /// convert and add to the BAZ
    /// 
    /// \return bool if slice was successfully added. False can be cause by
    ///              basecall and hfMetrics are not present or basecall does
    ///              not contains bases in production mode.
    bool AddZmwSlice(const Basecall* basecall, const uint16_t numEvents, 
                     std::vector<MetricBlock>&& hfMetrics, 
                     const uint32_t zmwId);

    /// Push internal BazBuffer to the queue of BazBuffers that are
    /// ready to be written to disk.
    /// 
    /// \return bool if data was available to be flushed out
    bool Flush();

The return bools are currently for debugging and will be repurposed to indicate
failures.

AddZmwSlice
^^^^^^^^^^^

AddZmwSlice() does not have to be called in the ascending order of the hole 
number. In fact, ZMW-slices can be added in random order. BazWriter takes care
of sorting them before they are written to disk.

Flush
^^^^^

Flush() can be called more than once per ZMW-slice. It has no effect on the final
output, but it has on the baz2bam performance, as more seeks as necessary for
stitching.

Buffering
^^^^^^^^^
BazWriter offers ONE internal BazBuffer to store the chunk that is currently 
appended to. In addition, BazWriter contains a queue of BazBuffers that wait 
to be written to disk. An independent thread is dedicated to take one BazBuffer 
at a time, compute SUPER_CHUNK_META+ZMW_SLICE_HEADER[] and write the chunk to disk.

CLI
^^^
BazWriter implements a demo CLI to simulate a BAZ::
    
	 bazwriter -h
  Usage: bazwriter [options]

  BazWriter simulates random base calls and writes a BAZ intermediate base file.

  Options:
    -h, --help            show this help message and exit

    Mandatory parameters:
      -o STRING           BAZ output filename

    Optional parameters:
      -v, --version       Print the tool version and exit
      -f STRING           Fasta input filename
      -F                  Adds an unsupported packet field (Forward compatibility test)
      -z INT              Number of ZMWs to simulate. Default: 1000000
      -b INT              Polymerase base pair rate per second. Default: 5
      -s INT              Chunk length in seconds. Default: 163
      -c INT              Chunks per movie. Default: 66
      -r                  Read?
      -p                  Internal pulse mode.
      -n                  No metrics mode.
      -q                  Bases without QVs/tags mode.
      -t                  Const data.
      --silent            No progress output.
      --sizeBasesWithoutQvs=INT
                          Output expected BAZ size for movie length in frames.
      --sizePulses=INT    Output expected BAZ size for movie length in frames.

CLI Examples
^^^^^^^^^^^^

Simulate a full production mode chip for three hours::

	$ bazwriter -o out.baz -z 1000000 -b 5 -s 163 -c 66

Simulate internal mode from a given fasta file. Sequences are chunked into 
163bp/s * 5s. Additional 30-35% pulses without base calls are simulated::

	$ bazwriter -o out.baz -f input.fasta -p

.. BazWriter Benchmarks
.. --------------------

.. **Background:** We assume a 100% loaded chip with one million ZMWS, a three hour
.. movie, and a polymerase base rate of 5 bp/s.

.. Goal: Compute the C_META and C_HEAD and write C_META, C_HEAD, BASE[], and HQ[] to disk below 120 seconds.

.. Setup: To simulate a sequel setting, we are performing in parallel:
..  - BazWriter: Writing 66 chunks with each one million ZMWs to run_2.baz
..  - Baz2bam: Reading run_1.baz, write each ZMW to BAM, FASTA.GZ, and FASTQ.GZ
..  - Bam2Customer: Transfer a file via scp

.. Results: **TO BE RECOMPUTED AFTER LAST SPEC HAS BEEN IMPLEMENTED.**

BazReader.h API
---------------

API of BazReader.h::

  /// Reads file header and super chunk header for indexed access.
  BazReader(const std::string& fileName, bool silent = true);

  /// Returns true if more ZMWs are available for reading.
  bool HasNext();

  /// Provides the next slice of stitched ZMWs.
  std::vector<StitchedZmw> NextSlice();

  /// Parses and provides the file header from the file stream
  std::unique_ptr<FileHeader> ReadFileHeader();

  /// Returns reference to current file header
  std::unique_ptr<FileHeader>& Fileheader();

  /// Number of ZMWs
  uint32_t NumZMWs();

  // Provides the ZMW ids of the next slice.
  std::vector<uint32_t> NextZmwIds();

  // Pops next slice from the queue of ZMW slices.
  inline void SkipNextSlice()

Demo
^^^^

How to use BazReader in practice::

  BazReader reader(FILENAME);
  while (reader.HasNext())
  {
      std::vector<StitchedZmw> stitchedZmwVector = reader.NextSlice();
      
      // Distribute batches of StitchedZmws to threadpool

      for (const auto& singleZmw: stitchedZmwVector)
      {
          // Access a single packet field
          if (zmw.HasPacketField(PacketFieldName::READOUT))
              for (const auto& foo : zmw.PacketField(PacketFieldName::READOUT))
                  // Do something with foo

          // Access a single high-frequency metric field
          if (zmw.HasHFMField(MetricFieldName::PR_0))
              for (const auto& hff : zmw.HFMField(MetricFieldName::PR_0))
                  // Do something with hff

          // Access a single medium-frequency metric field
          if (zmw.HasMFMField(MetricFieldName::PKMID_G))
              for (const auto& mff : zmw.MFMField(MetricFieldName::PKMID_G))
                  // Do something with mff

          // Access a single low-frequency metric field
          if (zmw.HasLFMField(MetricFieldName::PULSE_WIDTH))
              for (const auto& lff : zmw.LFMField(MetricFieldName::PULSE_WIDTH))
                  // Do something with lff
      }
  }