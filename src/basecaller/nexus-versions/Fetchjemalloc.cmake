include(findpathirdparty)

set(jemalloc_VERSION 5.1.0-bid3 CACHE INTERNAL "jemalloc version")
findpathirdparty(jemalloc jemalloc ${jemalloc_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
