include(findpathirdparty)

set(boost_VERSION gcc8-1.73.0-bid6 CACHE INTERNAL "boost version")
findpathirdparty(boost boost ${boost_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
