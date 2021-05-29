cmake_minimum_required(VERSION 3.13)

add_library(libBazioEncoding STATIC
   EncodingParams.h
   EncodingParams.cpp
   FieldNames.h
   FieldAccessors.h
   FieldAccessors.cpp
   FieldSerializers.h
   FieldSerializers.cpp
   FieldTransforms.h
   FieldTransforms.cpp
   PulseToBaz.h
   PulseToBaz.cpp
)

include(pacbio-cplusplus-api)

target_link_libraries(libBazioEncoding
    PUBLIC
        libPacbioCplusplusApi
)

target_include_directories(libBazioEncoding
    PUBLIC
#remove this dep
${BAZIO_PROJECT_DIR}/basecaller
        ${BAZIO_PROJECT_DIR}
)

add_subdirectory(test)
