include(findpathirdparty)

set(hdf5_VERSION gcc8-1.8.13-bid4 CACHE INTERNAL "hdf5 version")
findpathirdparty(hdf5 hdf5 ${hdf5_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
