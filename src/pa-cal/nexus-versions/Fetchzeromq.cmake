include(findpathirdparty)

set(zmq_VERSION gcc8-3.2.6-bid4 CACHE INTERNAL "zmq version")
findpathirdparty(zeromq zmq ${zmq_VERSION} ${CMAKE_SYSTEM_PROCESSOR})
