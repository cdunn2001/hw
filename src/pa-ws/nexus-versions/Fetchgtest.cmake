include(findpathirdparty)

set(gtest_VERSION gcc8-1.10.0-bid6 CACHE INTERNAL "gtest version")
findpathirdparty(gtest gtest ${gtest_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
