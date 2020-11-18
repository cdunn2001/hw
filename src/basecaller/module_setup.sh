# this must be SOURCED, not executed
. /etc/profile.d/modules.sh

module purge

module use /pbi/dept/primary/modulefiles
module use /mnt/software/modulefiles
module load devtoolset/6
# module load convey/OpenHT/2.1.35
module load openht/git
module load composer_xe/2017.4.196
module load pacbio-devtools
module load patchelf
module load libusb/1.0.22
module load cmake/3.13.3
module load cuda/10.1.168_418.67
module load ninja/1.10.0

# to build the Wolverine Personality:
# module load vivado/2018.3

