include(findpathirdparty)

set(expect_VERSION gcc8-5.44.1.15-bid3 CACHE INTERNAL "expect version")
findpathirdparty(expect expect ${expect_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
