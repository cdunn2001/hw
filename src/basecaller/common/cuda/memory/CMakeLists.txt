
cmake_minimum_required(VERSION 3.8)

set(CPP_LIST
    AllocationViews.h
    DataManagerKey.h
    PinnedHostArray.h
    PinnedHostArray.cpp
    SmartDeviceAllocation.h
    SmartDeviceAllocation.cpp
    GpuAllocationPool.h
    GpuAllocationPool.cpp
    UnifiedCudaArray.h
)

set(CUDA_LIST
    AllocationViews.cuh
    AllocationViews.cu
    DeviceOnlyArray.cu
    DeviceOnlyArray.cuh
)

add_library(libMongoCommonCudaMemory
    ${CPP_LIST}
    ${CUDA_LIST}
)
