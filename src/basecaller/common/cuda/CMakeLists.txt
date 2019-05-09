
cmake_minimum_required(VERSION 3.13)

set(CPP_LIST
    CudaFunctionDecorators.h
    DisableBoostOnDevice.h
    PBCudaRuntime.h
    PBCudaRuntime.cpp
    # PBCudaRuntimeNoCuda.cpp
    memory/AllocationViews.h
    memory/DataManagerKey.h
    memory/GpuAllocationPool.h
    memory/GpuAllocationPool.cpp
    memory/PinnedHostArray.h
    memory/PinnedHostArray.cpp
    memory/SmartDeviceAllocation.h
    memory/SmartDeviceAllocation.cpp
    memory/UnifiedCudaArray.h
)

set(CUDA_LIST
    memory/AllocationViews.cuh
    memory/AllocationViews.cu
    memory/DeviceOnlyArray.cuh
    memory/DeviceOnlyArray.cu
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
