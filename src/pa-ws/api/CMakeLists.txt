#
# This cmake code creates a target (libPaWsDoc) that
# contains statically linked text files, such as the REST API markdown,
# and a tiny wrapper to access these text files as std::strings.
# 

# convert C++ code to HTML
add_custom_command(OUTPUT kes-paws-rest-api-swagger.html
    DEPENDS swagger-yaml-to-html.py apihelp.h doc_accessor_TEMPLATE.cpp ObjectTypes.h PostprimaryObject.h ProcessStatusObject.h SocketObject.h StorageObject.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    COMMAND make 
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/kes-paws-rest-api-swagger.html ${CMAKE_CURRENT_BINARY_DIR}
)

# create a linkable library that has the documentation contents embedded as strings.
add_custom_command(OUTPUT docsPaWsApi.o
    DEPENDS kes-paws-rest-api-swagger.html
    COMMAND ld -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/docsPaWsApi.o kes-paws-rest-api-swagger.html
    COMMAND objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents ${CMAKE_CURRENT_BINARY_DIR}/docsPaWsApi.o ${CMAKE_CURRENT_BINARY_DIR}/docsPaWsApi.o
)

# dynamically create the function that is used to access the embedded strings.
set(FILENAME kes_paws_rest_api_swagger_html)
configure_file(doc_accessor_TEMPLATE.cpp apihelp_generated.cpp)

add_library(libPaWsApi
    STATIC
    apihelp.h
    apihelp_generated.cpp
    docsPaWsApi.o 
)

target_include_directories(libPaWsApi
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

SET_SOURCE_FILES_PROPERTIES(
    docsPaWsApi.o
  PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true
)

SET_TARGET_PROPERTIES(
    libPaWsApi
  PROPERTIES
    LINKER_LANGUAGE C 
)
