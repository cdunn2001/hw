How to run the HQRM (High Quality Region Metrics) Work Flow
-----------------------------------------------------------

To run the HQRM, you need
* A 3.1.0 or newer *.subreads.bam and *.scraps.bam
* The reference (Ecoli, or whatever)
* the desired toolset

How to set up a run
===================

Create a csv file of 4 columns:

* path to directory that was used by Milhouse. Example::

  /pbi/collections/319/3190023/r54006_20160520_201229/1_A01

* name of the movie. Example::

  m54006_160520_203903

* name of the reference. Example::

  2kET_4_asym_polyA

Load the required modules. Need bam2bam, baz2bam and R on the path::

  module load basecaller/mainline R/3.2.1 smrtanalysis/mainline

If you want to analyze a "rerun", i.e. a subdirectory of the original Milhouse job, then you need to
set an environment variable.::

  export rerundir=Bug33294_try1

Then run the ``qsub.sh`` script with the name of the csv file as the argument::

  ./qsub.sh thing.csv


This is what happens:
1. qsub.sh calls setup.sh and creates a working directory
2. qsub.sh then goes to that working directory and runs 'make submit'
3. the 'submit' Makefile target calls 'make all' from a submitted queue job.
4. The stdout and stderr of the queue job will be written in the working directory, with
   suffices: .o* and .e* for output and error respectively.
5. ??? the baz file is created to scraps and subreads again
6. the subreads/scripts.bam is processed into _hq.hqregions/scraps.bam
7. the subreads/scripts.bam is processed into _nohq.subreads/scraps.bam
8. the python script summarizes the bam files
9. the R script creates the graphical PDF report



Example Walkthrough
===================

 $ module load basecaller/mainline R/3.2.1
 $ cat thing.csv
 /pbi/collections/319/3190023/r54006_20160520_201229/1_A01 m54006_160520_203903 2kET_4_asym_polyA
 $ ./qsub.sh thing.csv
 $ wait several dozen minutes. Check 'qstat' to see if job is still running.
 $ view *.pdf


Setup on new machine:
---------------------

Install R

    yum install R


Batch running
-------------

Copy batch_template.sh to a new file and edit. You must run the script on a computer that supports the qsub command. For 
example, login14-biofx01. 

Each line of the batch job will appear in your $HOME/hqrm directory. The first few lines have instructions on how to remote 
login to the machine where the actual job ran.

When the job is done, the output will appear in $HOME/hqrm.

Run the `qstat` command to see what jobs are running.


How to submit to qsub.sh
------------------------

Edit a fofn with 3 columns:

 run directory
 movie name
 reference

The reference name must match a canonical reference fasta file. See runs4.fofn for an examples.

call qsub.sh with the fofn as sole argument.

How to run locally, without using qsub.sh
-----------------------------------------


See 

    ./test.sh

for an example usage.

The output of the run is a PDF file.

To run a more complicate configuration, you can run './setup.sh' (with required arguments), then cd to the output
folder and edit the Makefile with the variables you want. For example, if you want to run a special baz2bam,
you can edit the baz2bam_exec variable in the Makefile.

By default the baz file is NOT regenerated. To regenerate a new baz file (i.e. with a new basecaller-console-app)
then set the "baz" variable to a local file name, i.e. "baz=foo.baz". Then it will be regenerated and used
in the rest of the analysis.


make targets
------------

copy - makes all final products locally and copies final PDF and CSV files to home directory

archive - makes all final products locally and copies ALL files to the home directory


A
