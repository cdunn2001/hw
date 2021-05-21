cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0048 NEW)

project(bazviewer)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")

    if(NOT TARGET bazviewer)
        add_executable(bazviewer
            bazviewer.cpp
            ${CMAKE_CURRENT_LIST_DIR}/../HDFMultiArrayIO.cpp
        )

        set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../.."
                ${CMAKE_MODULE_PATH})

        target_link_libraries(bazviewer
            libBazio
            libSequelCommon1
            SegFault
        )

    endif()

endif()
