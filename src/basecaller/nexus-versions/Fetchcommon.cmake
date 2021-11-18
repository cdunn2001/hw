include(finddep)

include(pacbio-build-type)

set(pacommon_VERSION 0.0.0.SNAPSHOT134595 CACHE INTERNAL "pa_common version")
finddep(pacbio.seq.pa pacbio-pa-common-mongo ${pacommon_VERSION} ${PB_COMPILER_ARCH})

set(pacommon_ROOT ${pacbio-pa-common-mongo_ROOT})

