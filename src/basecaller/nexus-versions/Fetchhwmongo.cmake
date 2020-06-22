# Please read the source code for "cmake-targets/pacbio-hw-mongo.cmake" and understand how these two scripts
# work together. This script is mostly for grabbing a particular artifact for a particular version, and the
# other script is a wrapper to keep dependencies easy to use.

include(finddep)

include(pacbio-build-type)

set(hwmongo_VERSION 0.0.1.SNAPSHOT100287 CACHE INTERNAL "hwmongo version")
finddep(pacbio.seq.pa pacbio-hw-mongo ${hwmongo_VERSION} ${PB_COMPILER_ARCH})

set(hwmongo_ROOT ${pacbio-hw-mongo_ROOT})

