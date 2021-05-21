HQRF Metrics generation
-----------------------

The following documentation describes the flow for obtaining metrics on the HQRF algorithm.

Quick start
===========

see the ppa/dev/test.sh script.


Script Details
==============

The highest level script is "qsub.sh" which will launch several jobs on the SGE server, using `qsub`.

The next level script is "run.sh" which runs one job. This in turn calls "setup.sh" and then changes directories to the temporary
directory that setup.sh creates and runs "make".

The next level scripts is "setup.sh" which creates a temporary directory, which contains a master Makefile, and fills in
several default values for the Makefile variables.


The prefered method of invoking the script is to set all environment variables before the script name, such as

::

    inputdir=/pbi/collections/315/3150113/r54007_20160226_225820/1_A01 rerundir=Sequel_3.0.17 run=m54007_160226_225834 ./run.sh



Flow Details
============

The flow originates from a trc.h5 file. From this, baz2bam is run on it to create the subreads/scraps bam files.
These files are then used for two parallel flows. bam2bam is run with the --nohq option to generate subreads that ignore the
original HQ regions. bam2bam is run a second time with the --hqregion option to generate "dummy" subreads that are
just the HQ regions themselves. This is a hack to extract the start/stop positions of the HQ regions.

Penultimately, a python script (using pbcore) extracts the data from the two separate flows and writes out a CSV file.
The CSV file uses "NA" to fill in empty pieces of information. 

Finally, an R script generates the plots (PDF) and summary table (CSV).
