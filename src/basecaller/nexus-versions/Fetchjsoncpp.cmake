include(findpathirdparty)

set(jsoncpp_VERSION gcc8-1.4.4-bid4 CACHE INTERNAL "jsoncpp version")
findpathirdparty(open-source-parsers jsoncpp ${jsoncpp_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
