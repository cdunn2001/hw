include(findpathirdparty)

set(hdf5_VERSION 1.8.13-bid3 CACHE INTERNAL "hdf5 version")
findpathirdparty(hdf5 hdf5 ${hdf5_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
