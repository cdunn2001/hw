include(findpathirdparty)

set(zmq_VERSION 3.2.6-bid3 CACHE INTERNAL "zmq version")
findpathirdparty(zeromq zmq ${zmq_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
