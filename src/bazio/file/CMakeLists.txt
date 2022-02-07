cmake_minimum_required(VERSION 3.13)

add_library(libBazioFile STATIC
    FileFooterSet.h
    FileFooterSet.cpp
    FileHeader.h
    FileHeader.cpp
    FileHeaderSet.h
    FileHeaderSet.cpp
    FileHeaderBuilder.h
    FileHeaderBuilder.cpp
    RunLength.h
    RunLength.cpp
    ZmwInfo.h
    ZmwInfo.cpp
)

include(pacbio-cplusplus-api)

target_link_libraries(libBazioFile
    PUBLIC
        libPacbioCplusplusApi
        libBazio # I want to destroy this dep
        libHalf # potentially temporary?
)

target_include_directories(libBazioFile
    PUBLIC
#remove this dep
${BAZIO_PROJECT_DIR}/basecaller
        ${BAZIO_PROJECT_DIR}
)

add_subdirectory(test)
