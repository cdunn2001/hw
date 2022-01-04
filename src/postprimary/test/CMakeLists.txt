
if(FAKE_ERROR)
    add_definitions(-DFAKE_ERROR)
endif()

include(pacbio-gtest)

add_subdirectory(unit)
add_subdirectory(cram)

# simbazwriter

add_library(libSimBazWriter STATIC
    SimulateConfigs.h
    SimulateConfigs.cpp
    SimulateWriteUtils.h
    SimulateWriteUtils.cpp
    Simulation.h
    Simulation.cpp
    SimulationFromFasta.h
    SimulationRNG.h
)

target_link_libraries(libSimBazWriter
    PUBLIC
        libBazio
        libSequelCommon1
        libZlib
        libMongoDataTypes
)

add_executable(simbazwriter
    simbazwriter.cpp
)

target_link_libraries(simbazwriter
    PRIVATE
        libSimBazWriter
)

# need access to the ipcat binary as well.  It's either in our depcache for normal builds,
# or deep in our build tree for local builds of pa-common.  Either way bump up a copy to
# here so the cram tests can use it
if (TARGET ipcat)
    add_custom_command(
        TARGET simbazwriter 
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ipcat> ${CMAKE_CURRENT_BINARY_DIR}/ipcat
    )
else()
    add_custom_command(
        TARGET simbazwriter 
        PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:PaCommon::ipcat> ${CMAKE_CURRENT_BINARY_DIR}/ipcat
    )
endif()

