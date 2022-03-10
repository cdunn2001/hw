include(findpathirdparty)

set(jsoncpp_VERSION 1.4.4-bid3 CACHE INTERNAL "jsoncpp version")
findpathirdparty(open-source-parsers jsoncpp ${jsoncpp_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
