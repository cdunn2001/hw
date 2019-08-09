# this must be SOURCED, not executed

module purge
# This is mainly for the benefit of bamboo, and
# assumes developers are not installing and loading
# alternate devtoolsets.  There is no way to "purge"
# an already loaded devtoolset, though loading
# the same one multiple times has no effect
#
# Note: ` || true` is requires so this script can be
# executed in bash shell with `-e`, as it calls
# (and handles) a function that may return 1
#source scl_source enable devtoolset-6 || true

. /etc/profile.d/modules.sh
module use /pbi/dept/primary/modulefiles
module use /mnt/software/modulefiles
module load devtoolset/6
module load convey/OpenHT/2.1.35
module load composer_xe/2017.4.196
module load pacbio-devtools
module load patchelf
module load libusb/1.0.22
module load cmake/3.13.3
module load cuda

# to build the Wolverine Personality:
# module load vivado/2018.3

