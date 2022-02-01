include(findpathirdparty)

set(boost_VERSION gcc8-1.58.0-bid5 CACHE INTERNAL "boost version")
findpathirdparty(boost boost ${boost_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
