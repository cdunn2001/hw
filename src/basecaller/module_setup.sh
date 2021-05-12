# this must be SOURCED, not executed
. /etc/profile.d/modules.sh

module purge

module use /pbi/dept/primary/modulefiles
module use /mnt/software/modulefiles
module load devtoolset/8
module load composer_xe/2017.4.196
module load pacbio-devtools
module load patchelf
module load libusb/1.0.22
module load cmake/3.20.2
module load ninja/1.10.0
module load cuda/11.4.0_470.42.01
module load git
