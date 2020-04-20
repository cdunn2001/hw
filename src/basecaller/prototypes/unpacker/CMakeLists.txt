project(CudaTemplate LANGUAGES CXX)
cmake_minimum_required(VERSION 3.8)

add_executable(unpacker
        unpacker.h
        unpacker.cpp
        main.cpp
)

target_include_directories(unpacker
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(unpacker
    PRIVATE
        libPacbioCplusplusApi
)
