include(findpathirdparty)

set(jemalloc_VERSION gcc8-5.1.0-bid4 CACHE INTERNAL "jemalloc version")
findpathirdparty(jemalloc jemalloc ${jemalloc_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
