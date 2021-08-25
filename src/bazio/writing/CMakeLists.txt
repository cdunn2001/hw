cmake_minimum_required(VERSION 3.20)

add_library(libBazioWriting STATIC
    BazAggregator.h
    BazAggregator.cpp
    BazBuffer.h
    BazBuffer.cpp
    BazWriter.h
    BazWriter.cpp
    GrowableArray.h
    GrowableArray.cpp
    MemoryBuffer.h
    MemoryBuffer.cpp
    PacketBuffer.h
    PacketBuffer.cpp
    PacketBufferManager.h
    PacketBufferManager.cpp
)

include(pacbio-cplusplus-api)

target_link_libraries(libBazioWriting
    PUBLIC
        libPacbioCplusplusApi
        libBazio # I want to destroy this dep
        libHalf # potentially temporary?
)

target_include_directories(libBazioWriting
    PUBLIC
#remove this dep
${BAZIO_PROJECT_DIR}/basecaller
        ${BAZIO_PROJECT_DIR}
)

add_subdirectory(test)
