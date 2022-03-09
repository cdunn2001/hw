include(findpathirdparty)

set(zlib_VERSION 1.2.8-bid3 CACHE INTERNAL "zlib version")
findpathirdparty(zlib zlib ${zlib_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
