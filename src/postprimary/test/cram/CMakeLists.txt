
add_test(NAME SequelPpaCram
    COMMAND ./runcram.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
)

add_executable(stsTool
    stsTool.cpp
)

target_link_libraries(stsTool
    libSequelCommon
)
