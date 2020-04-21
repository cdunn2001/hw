project(CudaTemplate LANGUAGES CXX)
cmake_minimum_required(VERSION 3.8)

if(${CMAKE_CXX_COMPILER} MATCHES "[cg]\\+\\+")
    # this is just a test. It doesn't need to be compiled. I had a problem
    # enabling SSE4 for icc, so I'm only compiling for gcc (g++).
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
endif()