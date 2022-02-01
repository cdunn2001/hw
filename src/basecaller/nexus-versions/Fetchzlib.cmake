include(findpathirdparty)

set(zlib_VERSION gcc8-1.2.8-bid4 CACHE INTERNAL "zlib version")
findpathirdparty(zlib zlib ${zlib_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
