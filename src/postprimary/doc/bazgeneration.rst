Generate BAM from TRC
#####################

.. highlight:: sh

Imagine you have a full-chip or 32k subsampled trace file from an old 
PrimaryAnalysis run, but want to test the latest and greatest basecaller and PPA
on it. Good news, it's easier than you think.

First step, login onto a secondary node::

    ssh login14-biofx01

And enter a machine on the grid, so you don't flood the login node::

    qrsh

Load the appropriate module::
    
    module load basecaller/3.1.1-current

Change directory where the trace file is stored. Save the movie name in $MOVIE, 
create a new analysis folder, and produce the new BAZ::

    MOVIE=m54012_160622_092039
    mkdir Sequel_3.1.1 && cd Sequel_3.1.1
    basecaller-console-app --inputfile ../${MOVIE}.trc.h5 \
                           --outputbazfile ${MOVIE}.baz

This will give you following BAZ file::

    m54012_160622_092039.baz

Now, we need to convert the BAZ file to BAM; the correct PPA version
has already been loaded with the basecaller module::

    baz2bam ${MOVIE}.baz -o ${MOVIE} -j 12 -b 12 \
                         -m ../${MOVIE}.run.metadata.xml \
                         --adapter ../${MOVIE}.adapters.fasta 

We get following additional, customer facing, files::

    m54012_160622_092039.scraps.bam
    m54012_160622_092039.scraps.bam.pbi
    m54012_160622_092039.sts.xml
    m54012_160622_092039.subreads.bam
    m54012_160622_092039.subreads.bam.pbi
    m54012_160622_092039.subreadset.xml

