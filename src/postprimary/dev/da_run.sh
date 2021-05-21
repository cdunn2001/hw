
export basecaller_console_app_exec=/home/UNIXHOME/mlakata/p4/basecaller-dha/Sequel/basecaller/build/x86_64/Release/application/basecaller-console-app
export bam2bam_exec=/home/UNIXHOME/mlakata/p4/basecaller-dha/Sequel/ppa/build/x86_64/Release/bam2bam
export baz2bam_exec=/home/UNIXHOME/mlakata/p4/basecaller-dha/Sequel/ppa/build/x86_64/Release/baz2bam
export reportdir=/home/UNIXHOME/mlakata/hqrm_da
export baz=rerun.baz

. /etc/profile.d/modules.sh
module load basecaller/3.0.17 # get so libraries on path
module load smrtanalysis/mainline


./qsub.sh runs4.csv
