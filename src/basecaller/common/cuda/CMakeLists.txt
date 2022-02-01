
cmake_minimum_required(VERSION 3.20)

set(CPP_LIST
    CudaFunctionDecorators.h
    DisableBoostOnDevice.h
    CudaLaneArray.cu
    CudaLaneArray.cuh
    PBCudaSimd.h
    PBCudaRuntime.h
    PBCudaRuntime.cpp
    # PBCudaRuntimeNoCuda.cpp
    memory/AllocationViews.h
    memory/DataManagerKey.h
    memory/DeviceAllocationStash.h
    memory/DeviceAllocationStash.cpp
    memory/ManagedAllocations.h
    memory/ManagedAllocations.cpp
    memory/SmartDeviceAllocation.h
    memory/SmartDeviceAllocation.cpp
    memory/StashableDeviceAllocation.h
    memory/StashableDeviceAllocation.cpp
    memory/UnifiedCudaArray.h
    streams/CudaEvent.h
    streams/CudaEvent.cpp
    streams/CudaStream.h
    streams/CudaStream.cpp
    streams/KernelLaunchInfo.h
    streams/KernelLaunchInfo.cpp
    streams/StreamMonitors.h
    streams/StreamMonitors.cpp
)

set(CUDA_LIST
    PBCudaSimd.cu
    PBCudaSimd.cuh
    memory/AllocationViews.cuh
    memory/AllocationViews.cu
    memory/DeviceOnlyArray.cuh
    memory/DeviceOnlyArray.cu
    memory/DeviceOnlyObject.cuh
    memory/DeviceOnlyObject.cu
    streams/LaunchManager.cuh
    utility/CudaArray.h
    utility/CudaArray.cu
    utility/CudaTuple.h
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

