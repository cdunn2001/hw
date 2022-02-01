cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0048 NEW)

project(bazviewer)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")

    if(NOT TARGET bazviewer)
        add_executable(bazviewer
            bazviewer.cpp
        )

        target_link_libraries(bazviewer
            libBazio
            libSequelCommon1
            SegFault
        )

    endif()

endif()
