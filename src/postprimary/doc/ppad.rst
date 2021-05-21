Post-Primary Analysis Daemon
############################

The Post-Primary Analysis Daemon (PPAd) is an on-instrument service that is
responsible to handle communication between PAWS and baz2bam.

Daemon as a feature
===================

There is need for a separate daemon that invokes baz2bam. The main reason is
that baz2bam, as its counterpart bam2bam, has been designed to be a
standalone command-line application. Its self-suffiency allows to it be used on 
any machine, in any workflow. Unlike the primary RT-pipeline and ConsoleApp,
baz2bam is guaranteed to reproduce the exact results no matter where used.

To avoid a tight coupling with PAWS and maintain an extra daemon
layer within baz2bam, PPAd has been created.

Versioning and packaging
========================

This daemon is packaged together with baz2bam in the rpm that is being 
deployed on the instrument. This tight coupling avoids confusion what baz2bam
version is getting invoked. The PPAd version is the same as the PPA version
that is hardcoded in the Sequel/ppa/ppaToolboxVersion.txt file.

Systemd
=======

PPAd gets automatically started by the pacbio-pa service. Its name is
pacbio-ppad-*version*.service, e.g., pacbio-ppad-2.5.4.service

Service Configuration
---------------------

The systemd service (pacbio-ppad-\*.service) loads configuration information
from /etc/pacbio/pacbio-ppa.conf. This file sets two environment variables

    PPA_CONF=
    PPA_BAZ2BAM_OPTIONS=

PPA_CONF sets the additional command line arguments to ppad. PPA_BAZ2BAM_OPTIONS sets
additional command line arguments to baz2bam.

Example::

    PPA_CONF=--reducedstatsconfig /my/dir/rstconf.json


IPC Interfaces
==============

PPAd communicates via JSON messages over IPC:
 - PPAd listens on port 5600 for commands from PAWS.
 - PPAd sends status updates on port 5601 to PAWS.
 - PPAd listens on port 46661 for status updates of baz2bam.

PAWS => PPAd Interface
----------------------

PAWS can send three types of messages: **ppaStart**, **ppaStop**, **abort**"

Message ppaStart
^^^^^^^^^^^^^^^^

This command starts a new baz2bam process.
The message has following mandatory fields:

 - **acqId**, the acquisition ID in form of an UUID
 - **outputPrefix**, the prefix of the output files, commonly the movie name
 - **bazFile**, the absolute path to the input BAZ file
 - **rmdFile**, the absolute path to the run.metadata.xml file

Example::

    {
        "acqId"        : "ae4eac72-0655-11e6-b512-3e1d05defe78",
        "outputPrefix" : "m54011_160419_040029",
        "bazFile"      : "/data/pa/m54011_160419_040029.baz",
        "rmdFile"      : "/data/pa/.m54011_160419_040029.run.metadata.xml"
    }

Optional fields are:

 - **outputFolder**, the absolute path to the output directory,  defaults to "/data/pa" if not set
 - **controlsFile**, the absolute path to the controls fasta file,  defaults to "/etc/pacbio/600bp_Control_c2.fasta" if not set.  Control filtering can be deactivated by setting to an empty string ""
 - **computingThreads**, number of computing threads for baz2bam, defaults to 12 if not set
 - **bamThreads**, number of bam compression threads, defaults to 12 if not set
 - **minSubLength**, minimal subread length, defaults to 50 if not set
 - **minSnr**, minimal SNR, defaults to  4 if not set
 - **barcodesFile** and **barcodeScoreMode**, if both are set barcoding will be  performed. "barcodesFile" is the absolute path to the fasta file containing the barcodes. "barcodeScoreMode" is either "symmetric" or "asymmetric".

Example::

    {
        "acqId"            : "ae4eac72-0655-11e6-b512-3e1d05defe78",
        "outputPrefix"     : "m54011_160419_040029",
        "bazFile"          : "/data/pa/m54011_160419_040029.baz",
        "rmdFile"          : "/data/pa/.m54011_160419_040029.run.metadata.xml"
        "outputFolder"     : "/data/pb/",
        "controlsFile"     : "",
        "computingThreads" : 15,
        "bamThreads"       : 8,
        "minSubLength"     : 100,
        "minSnr"           : 2.5,
        "barcodesFile"     : "/my/path/96set.fasta",
        "barcodeScoreMode" : "asymmetric"
    }


Message ppaStop
^^^^^^^^^^^^^^^

This command stops the currently running baz2bam process that has the same 
acquisition id. If this id does not match the running process, an error will
be logged, but the baz2bam process will be stopped nevertheless.
This message has one mandatory field, **acqId**. Example::

    {
        "acqId" : "ae4eac72-0655-11e6-b512-3e1d05defe78"
    }

Message abort
^^^^^^^^^^^^^

This command stops baz2bam and aborts shuts down PPAd.
This message has one mandatory field, **acqId**. Example::

    {
        "acqId" : "ae4eac72-0655-11e6-b512-3e1d05defe78"
    }


PPAd => PAWS Interface
----------------------

PPAd reports back to PAWS via status messages that contains four fields:

 - **acqId**, the original acquisition ID in form of an UUID
 - **progress**, a number between 0 and 100
 - **state**, one of the types specified below
 - **message**, possible additional information

Possible states:

 - complete
 - start
 - idle
 - busy
 - progress
 - error
 - warning
   
Errors
^^^^^^

In the case of an unrecoverable error that terminates baz2bam, 
baz2bam sends a "ppa/error" to PPAd, including the acqId and the message with
error details. This particular message gets forwarded to PAWS and the listener
to baz2bam is closed.

An overview of possible errors, not complete:

 - **TRUNCATED_BAM**, whereas at least one output bam is corrupt
 - **ABORTED_BY_PAWS**, in case PAWS requested a termination
 - **HQ_METRICS_MISSING**, metrics required to compute the HQ region are not available in this BAZ file
 - **INVALID_INPUT_FILE**, input file does not exist
 - **INPUT_FILE_IS_DIRECTORY**
 - **INVALID_SEQUENCING_CHEMISTRY_EXCEPTION**, unsupported sequencing chemistry combination

Proposed FOX interface changes
==============================

To simplify the PAWS => PPAd interface, the JSON message could be slimmed down to:

 - **acqId**, the acquisition ID in form of an UUID
 - **movieName**, the movie name that is the prefix of the BAZ file
 - **folder**, the absolute folder path to the BAZ file, also used for output files

Example::

    {
        "acqId"     : "ae4eac72-0655-11e6-b512-3e1d05defe78",
        "movieName" : "m54011_160419_040029",
        "folder"    : "/data/pa/"
    }
