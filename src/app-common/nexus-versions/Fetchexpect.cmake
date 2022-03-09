include(findpathirdparty)

set(expect_VERSION 5.44.1.15-bid2 CACHE INTERNAL "expect version")
findpathirdparty(expect expect ${expect_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
