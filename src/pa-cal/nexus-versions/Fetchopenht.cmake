# Please read the source code for pa-third-party "cmake-targets/pacbio-openht.cmake" and understand how these two scripts
# work together. This script is mostly for grabbing a particular artifact for a particular version, and the
# other script is a wrapper to keep dependencies easy to use.

include(finddep)

include(pacbio-build-type)

set(openht_VERSION 0.0.1.SNAPSHOT117818 CACHE INTERNAL "openht version")
finddep(pacbio.seq.pa pacbio-openht ${openht_VERSION} ${PB_COMPILER_ARCH})

set(openht_ROOT ${pacbio-openht_ROOT})

