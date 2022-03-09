include(findpathirdparty)

set(boost_VERSION 1.58.0-bid3 CACHE INTERNAL "boost version")
findpathirdparty(boost boost ${boost_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
