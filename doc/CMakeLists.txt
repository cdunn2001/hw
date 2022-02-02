#
# This cmake code creates a target (libPaWsDoc) that
# contains statically linked text files, such as the REST API markdown,
# and a tiny wrapper to access these text files as std::strings.
#

# convert Markdown to HTML
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kes-paws-rest-api.html
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/kes-paws-rest-api.md ${CMAKE_CURRENT_LIST_DIR}/render_md2html.sh
    COMMAND ${CMAKE_CURRENT_LIST_DIR}/render_md2html.sh ${CMAKE_CURRENT_LIST_DIR}/kes-paws-rest-api.md kes-paws-rest-api.html
)

# create a linkable library that has the documentation contents embedded as
# strings.
add_custom_command(OUTPUT docsPaWs.o
    DEPENDS kes-paws-rest-api.html
    COMMAND ld -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/docsPaWs.o kes-paws-rest-api.html
    COMMAND objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents ${CMAKE_CURRENT_BINARY_DIR}/docsPaWs.o ${CMAKE_CURRENT_BINARY_DIR}/docsPaWs.o
)

# dynamically create the function that is used to access the embedded strings.
set(FILENAME kes_paws_rest_api_html)
configure_file(doc_accessor_TEMPLATE.cpp dochelp_generated.cpp)

add_library(libPaWsDoc
    STATIC
    dochelp.h
    dochelp_generated.cpp
    docsPaWs.o 
)

target_include_directories(libPaWsDoc
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

SET_SOURCE_FILES_PROPERTIES(
    docsPaWs.o
  PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true
)

SET_TARGET_PROPERTIES(
    libPaWsDoc
  PROPERTIES
    LINKER_LANGUAGE C 
)


find_package(Doxygen OPTIONAL_COMPONENTS dot)
if (DOXYGEN_FOUND)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")
    doxygen_add_docs(docHwMongo
        ${CMAKE_SOURCE_DIR}
    )
endif(DOXYGEN_FOUND)

#add_custom_command(
#    
# )BUILD=build
# PUBLISH=/pbi/dept/primary/docs/hw-mongo

# all : $(BUILD)/index.html $(BUILD)/kes-personality-requirements.html $(BUILD)/kes-paws-rest-api.html $(BUILD)/kes-paws.html

# install: all
# 	sudo mkdir -p $(PUBLISH)
# 	sudo cp $(BUILD)/* $(PUBLISH)
# 	sudo cp *.png *.jpg *.svg $(PUBLISH)

# rundoxygen: Doxyfile
# 	doxygen Doxyfile

# Within Confluence, click meatball menu, View Storage Format
# Copy and paste the storage format to xml
# At command line:
#   pandoc -i test.xml -o test.md -f html -t markdown_github
