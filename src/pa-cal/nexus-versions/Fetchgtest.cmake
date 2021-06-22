include(findpathirdparty)

set(gtest_VERSION 1.10.0-bid4 CACHE INTERNAL "gtest version")
findpathirdparty(gtest gtest ${gtest_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
