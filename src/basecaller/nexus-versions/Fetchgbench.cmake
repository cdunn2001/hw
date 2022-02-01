include(findpathirdparty)

set(gbench_VERSION gcc8-1.5.6-bid4 CACHE INTERNAL "gbench version")
findpathirdparty(gbench gbench ${gbench_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
