#
# TBB configuration
#

if(NOT ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "k1om")
  find_path(TBB_ROOT include PATHS /opt/intel/compilers_and_libraries_2017.4.196/linux/tbb /opt/intel/tbb "C:/Program Files/tbb43_20150209oss" NO_DEFAULT_PATH)
  message(STATUS "TBB_ROOT = ${TBB_ROOT}")

  if (TBB_ROOT)
    include_directories(SYSTEM ${TBB_ROOT}/include)

    find_library(TBB_LIBRARY             tbb              ${TBB_ROOT}/lib/intel64/gcc4.4  NO_DEFAULT_PATH )
    find_library(TBB_DEBUG_LIBRARY       tbb_debug        ${TBB_ROOT}/lib/intel64/gcc4.4  NO_DEFAULT_PATH )
    find_library(TBBMALLOC_LIBRARY       tbbmalloc        ${TBB_ROOT}/lib/intel64/gcc4.4  NO_DEFAULT_PATH )
    find_library(TBBMALLOC_DEBUG_LIBRARY tbbmalloc_debug  ${TBB_ROOT}/lib/intel64/gcc4.4  NO_DEFAULT_PATH )
  else (TBB_ROOT)
    # No TBB_ROOT. Try looking for an open-source RPM install.
    # Assume that /usr/include is a standard system include path.
    find_library(TBB_LIBRARY             tbb              /usr/lib64 )
    find_library(TBB_DEBUG_LIBRARY       tbb_debug        /usr/lib64 )
    find_library(TBBMALLOC_LIBRARY       tbbmalloc        /usr/lib64 )
    find_library(TBBMALLOC_DEBUG_LIBRARY tbbmalloc_debug  /usr/lib64 )
  endif (TBB_ROOT)

  find_library(TBB_LIBRARY             tbb              "C:/Program Files/tbb43_20150209oss/lib/intel64/vc12")
  find_library(TBB_DEBUG_LIBRARY       tbb_debug        "C:/Program Files/tbb43_20150209oss/lib/intel64/vc12")
  find_library(TBBMALLOC_LIBRARY       tbbmalloc        "C:/Program Files/tbb43_20150209oss/lib/intel64/vc12")
  find_library(TBBMALLOC_DEBUG_LIBRARY tbbmalloc_debug  "C:/Program Files/tbb43_20150209oss/lib/intel64/vc12")


  if(NOT TARGET tbb )
    add_library(tbb             UNKNOWN IMPORTED)
    add_library(tbb_debug       UNKNOWN IMPORTED)
    add_library(tbbmalloc       UNKNOWN IMPORTED)
    add_library(tbbmalloc_debug UNKNOWN IMPORTED)

    set_target_properties(tbb              PROPERTIES    IMPORTED_LOCATION ${TBB_LIBRARY}  INSTALL_RPATH_USE_LINK_PATH true  INSTALL_RPATH /zfadsfs BUILD_WITH_INSTALL_RPATH true)
    list(APPEND INSTALL_RPATH  ${TBB_ROOT}/lib/intel64/gcc4.4)

    set_target_properties(tbb_debug        PROPERTIES    IMPORTED_LOCATION ${TBB_DEBUG_LIBRARY}  )
    set_target_properties(tbbmalloc        PROPERTIES    IMPORTED_LOCATION ${TBBMALLOC_LIBRARY}  )
    set_target_properties(tbbmalloc_debug  PROPERTIES    IMPORTED_LOCATION ${TBBMALLOC_DEBUG_LIBRARY}  )
  
    set_property(TARGET tbb    PROPERTY IMPORTED_LINK_INTERFACE_LIBRARIES  tbb_debug tbbmalloc_debug)
  endif()


  if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
      get_filename_component(TBB_LIB_DIR ${TBB_LIBRARY} PATH)
      set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} ${TBB_LIB_DIR})
      message(STATUS "TBB_LIB_DIR = ${TBB_LIB_DIR}, ${TBB_LIBRARY}")

      #set_property(TARGET tbb PROPERTY LINK_FLAGS -L${TBB_LIB_DIR})  # doesn't work
      #set( ENV{LIBRARY_PATH} ${TBB_LIB_DIR} $ENV{LIBRARY_PATH}})     # doesn't work
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${TBB_LIB_DIR}") # works!
    endif()
  endif()
else()
  message(WARN "TBB configuration skipped, k1om")
endif()

if(NOT TARGET libTbb)
  add_library(libTbb INTERFACE)
  target_link_libraries(libTbb INTERFACE tbb)
  target_include_directories(libTbb SYSTEM INTERFACE ${TBB_ROOT}/include)
endif()
