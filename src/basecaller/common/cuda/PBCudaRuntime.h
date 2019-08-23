#ifndef PACBIO_CUDA_PB_CUDA_RUNTIME_H
#define PACBIO_CUDA_PB_CUDA_RUNTIME_H

#include <driver_types.h>

#include <cstdlib>

// This file serves two purposes.  The first is to be a firewall between the
// rest of our code and the cuda_runtime, to avoid including their large
// header everywhere.  The other to to help enforce good error checking.
// Each one of these functions will automatically check the error codes and
// squawk if there was a problem

namespace PacBio {
namespace Cuda {

size_t RequiredRegisterCount(const void* func);
size_t AvailableRegistersPerBlock();

cudaEvent_t InitializeEvent();
void DestroyEvent(cudaEvent_t event);
void RecordEvent(cudaEvent_t event);
bool CompletedEvent(cudaEvent_t event);
void SyncEvent(cudaEvent_t event);

void* CudaRawMalloc(size_t size);
void* CudaRawMallocHost(size_t size);
void* CudaRawMallocManaged(size_t size);
void* CudaRawMallocHostZero(size_t size);
void* CudaRawHostGetDevicePtr(void* p);
void  CudaFree(void* t);
void  CudaFreeHost(void* t);
void  CudaRawCopyHost(void* src, void* dest, size_t size);
void  CudaRawCopyDevice(void* src, void* dest, size_t size);
void  CudaSynchronizeDefaultStream();

template <typename T>
T* CudaMalloc(size_t count) { return static_cast<T*>(CudaRawMalloc(count*sizeof(T))); }
template <typename T>
T* CudaMallocHost(size_t count) { return static_cast<T*>(CudaRawMallocHost(count*sizeof(T))); }
template <typename T>
T* CudaMallocHostZero(size_t count) { return static_cast<T*>(CudaRawMallocHostZero(count*sizeof(T))); }
template <typename T>
T* CudaHostGetDevicePtr(T* p) { return static_cast<T*>(CudaRawHostGetDevicePtr(p)); }
template <typename T>
T* CudaMallocManaged(size_t count) { return static_cast<T*>(CudaRawMallocManaged(count*sizeof(T))); }
template <typename T>
void CudaCopyHost(T* dest, T* src, size_t count) { CudaRawCopyHost(dest, src, count*sizeof(T)); }
template <typename T>
void CudaCopyDevice(T* dest, T* src, size_t count) { CudaRawCopyDevice(dest, src, count*sizeof(T)); }

}} // ::Pacbio::Cuda

#endif //PACBIO_CUDA_PB_CUDA_RUNTIME_H
