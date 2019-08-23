
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    CudaFunctionDecorators.h
    DisableBoostOnDevice.h
    PBCudaSimd.h
    PBCudaSimd.cu
    PBCudaSimd.cuh
    PBCudaRuntime.h
    PBCudaRuntime.cpp
    # PBCudaRuntimeNoCuda.cpp
    memory/AllocationViews.h
    memory/DataManagerKey.h
    memory/AllocationPool.h
    memory/AllocationPool.cpp
    memory/SmartHostAllocation.h
    memory/SmartHostAllocation.cpp
    memory/SmartDeviceAllocation.h
    memory/SmartDeviceAllocation.cpp
    memory/UnifiedCudaArray.h
    streams/CudaEvent.h
    streams/CudaEvent.cpp
    streams/KernelLaunchInfo.h
    streams/KernelLaunchInfo.cpp
    streams/StreamMonitors.h
    streams/StreamMonitors.cpp
)

set(CUDA_LIST
    memory/AllocationViews.cuh
    memory/AllocationViews.cu
    memory/DeviceOnlyArray.cuh
    memory/DeviceOnlyArray.cu
    memory/DeviceOnlyObject.cuh
    memory/DeviceOnlyObject.cu
    streams/LaunchManager.cuh
    utility/CudaArray.h
    utility/CudaArray.cu
    utility/CudaTuple.cuh
    utility/CudaTuple.cu
)

add_library(libMongoCommonCuda STATIC
   ${CPP_LIST}
   ${CUDA_LIST}
)

target_link_libraries(libMongoCommonCuda
    PUBLIC
        libPacbioCplusplusApi
)

target_include_directories(libMongoCommonCuda
    PUBLIC 
        "${MONGO_PROJECT_DIR}"
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set_target_properties(libMongoCommonCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(test)

