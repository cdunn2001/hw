include(finddep)

set(pbbam_VERSION 1.6.1.SNAPSHOT109868 CACHE INTERNAL "pbbam version")
finddep(pacbio.seq.pa pbbam ${pbbam_VERSION} x86_64)

