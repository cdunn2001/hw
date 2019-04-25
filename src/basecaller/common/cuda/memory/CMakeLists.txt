
cmake_minimum_required(VERSION 3.13)

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

add_library(libMongoCommonCudaMemory STATIC
    ${CPP_LIST}
    # ${CUDA_LIST}
)
