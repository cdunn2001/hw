#
# This cmake code creates a target (libPaWsDashboard) that
# contains staticly linked text files, such as the REST API markdown,
# and a tiny wrapper to access these text files as std::strings.
#

add_custom_command(OUTPUT dashboard.o
    DEPENDS dashboard.html
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && ld -r -b binary -o ${CMAKE_CURRENT_BINARY_DIR}/dashboard.o dashboard.html
    COMMAND objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents 
         ${CMAKE_CURRENT_BINARY_DIR}/dashboard.o ${CMAKE_CURRENT_BINARY_DIR}/dashboard.o
)

add_library(libPaWsDashboard
    STATIC
    DashboardWrapper.h
    dashboard.o 
)
target_include_directories(libPaWsDashboard
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

SET_SOURCE_FILES_PROPERTIES(
    dashboard.o
    PROPERTIES
    EXTERNAL_OBJECT true
    GENERATED true
)

SET_TARGET_PROPERTIES(
    libPaWsDashboard
    PROPERTIES
    LINKER_LANGUAGE C 
)
