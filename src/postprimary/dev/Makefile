SHELL=/bin/bash

include Makefile.settings

.SHELLFLAGS=-lc
# inputdir=/pbi/collections/315/3150113/r54007_20160226_225820/1_A01
# movie=m54007_160226_225834
# rerundir=Sequel_3.0.17

#ref=/mnt/secondary/iSmrtanalysis/current/common/references/ecoliK12_pbi_March2013/sequence/ecoliK12_pbi_March2013.fasta
reference?=ecoliK12_pbi_March2013
ref?=/mnt/secondary/iSmrtanalysis/current/common/references/$(reference)/sequence/$(reference).fasta
inputs= $(movie).subreads.bam $(movie).scraps.bam
runxml=$(inputdir)/.$(movie).run.metadata.xml
baz?=$(inputdir)/$(rerundir)/$(movie).baz
trc?=$(inputdir)/$(movie).trc.h5
VE=/home/UNIXHOME/mlakata/git/PRmm/VE
bam2bam_exec?=bam2bam
baz2bam_exec?=baz2bam
basecaller_console_app_exec?=basecaller-console-app
reportdir?=$(HOME)/hqrm
archivedir?=$(reportdir)/$(movie)

all: copy

copy: $(reportdir)/$(movie).pdf $(reportdir)/$(movie)_metrics.csv

pdf: $(movie).pdf

$(reportdir)/% : %
	cp -p $< $@

poly: $(movie)_poly.zmws.bam

csv: $(movie).csv

execs:
	@echo bam2bam is $(bam2bam_exec)
	@echo baz2bam is $(baz2bam_exec)
	@echo basecaller-console-app is $(basecaller-console-app_exec)
	@echo trc is $(trc)
	@echo baz is $(baz)

####################################################

sanity:
	# bam2bam --version 2>&1 | gawk -F. '{print $4}'
	# 176147

$(movie).pdf $(movie)_metrics.csv: $(movie).csv plots.r
	Rscript --verbose plots.r $< $@ $(movie)_metrics.csv
	echo PDF in $(outputdir)/$@

$(movie).csv: $(movie)_hq.hqregions.bam $(movie)_nohq.aligned.bam $(movie)_poly.zmws.bam script2.py
	source $(VE)/bin/activate && ./script2.py $(movie)_hq.hqregions.bam $(movie)_nohq.aligned.bam $(movie)_poly.zmws.bam > $@ || rm -f $@

$(movie)_hq.hqregions.bam: $(inputs)
	$(bam2bam_exec) -j12 -b12 --minSnr=0 --hqregion $(inputs) -o $(movie)_hq

$(movie)_nohq.aligned.bam: $(movie)_nohq.subreads.bam
	pbalign --algorithm blasr $< $(ref) $@

$(movie)_nohq.subreads.bam:  $(inputs)
	$(bam2bam_exec) -j12 -b12 --fullHQ --adapters ./adapters.fasta -o $(movie)_nohq $(inputs)

$(movie)_nohq.aligned.txt: $(movie)_nohq.subreads.bam
	blasr -m 4 $< $(ref) > $@

$(movie)_poly.zmws.bam $(movie)_poly.scraps.bam:  $(inputs)
	$(bam2bam_exec) --minSnr=0 --polymerase -o $(outputdir)/$(movie)_poly $(inputs)

$(movie).subreads.bam $(movie).scraps.bam: $(baz) adapters.fasta 600bp_Control_c2.fasta $(runxml)
	$(baz2bam_exec) $< -o $(movie) --minSnr=0 --metadata $(runxml) -j 12 -b 4 --adapters ./adapters.fasta --controls ./600bp_Control_c2.fasta

$(baz): $(trc)
	$(basecaller_console_app_exec) --internal --input $< --output $@


#adapters.fasta: $(runxml)
#	echo ">smrt" > $@
#   xmllint --xpath "//*[local-name()='RightAdaptorSequence']/text()" $< >> $@

adapters.fasta: $(inputdir)/$(movie).adapters.fasta
	cp $< $@

600bp_Control_c2.fasta:
	cp $(srccode)/600bp_Control_c2.fasta .

show:
	samtools view $(inputdir)/$(rerundir)/$(movie).subreads.bam | cut -c1-100
	samtools view $(movie)_hq.hqregions.bam | cut -c1-100
	samtools view $(movie)_nohq.subreads.bam | cut -c1-100
	samtools view $(movie)_nohq.aligned.bam | cut -c1-100 | sort

clean:
	rm -f *.bam *.bam.pbi *.subreadset.xml *.sts.xml *.bam.tmp *.log adapters.fasta *.txt

install:
	Rscript install.r

update: plots.r script2.py Makefile

plots.r: $(srccode)/plots.r
	chmod 644 $@ || true
	cp $< $@

script2.py: $(srccode)/script2.py
	chmod 755 $@ || true
	cp $< $@
	chmod 755 $@

Makefile: $(srccode)/Makefile
	chmod 644 $@ || true
	cp $< $@

archive:
	mkdir -p $(archivedir)
	rsync * $(archivedir)

submit:
	qsub -q sequel-farm -pe smp 12 -V -cwd -b y -N hqrm_$(movie) -o . -e . make all
	 

