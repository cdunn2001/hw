#ifndef PACBIO_CUDA_PB_CUDA_RUNTIME_H
#define PACBIO_CUDA_PB_CUDA_RUNTIME_H

#include <driver_types.h>

#include <cstdlib>
#include <vector>
#include <string>

#include <pacbio/PBException.h>

// This file serves two purposes.  The first is to be a firewall between the
// rest of our code and the cuda_runtime, to avoid including their large
// header everywhere.  The other to to help enforce good error checking.
// Each one of these functions will automatically check the error codes and
// squawk if there was a problem

namespace PacBio {
namespace Cuda {

struct CudaMemException : public std::runtime_error {
    CudaMemException(const std::string& str)
        : std::runtime_error{str}
    {}
};
// Should probably elevate this to pa-common at some point
#define PBExceptionCust(msg, Type)    PacBio::PBExceptionEx<Type>(msg,__FILE__,__LINE__, __FUNCTION__, PB_TRACEBACK() );

size_t RequiredRegisterCount(const void* func);
size_t AvailableRegistersPerBlock();

cudaEvent_t InitializeEvent();
void DestroyEvent(cudaEvent_t event);
void RecordEvent(cudaEvent_t event);
bool CompletedEvent(cudaEvent_t event);
void SyncEvent(cudaEvent_t event);

struct SupportedStreamPriorities
{
    // Note: Numerically, lower integer values
    //       are *higher* priority
    int leastPriority;
    int greatestPriority;
};
SupportedStreamPriorities StreamPriorityRange();
cudaStream_t CreateStream(int priority);
void DestroyStream(cudaStream_t stream);

void* CudaRawMalloc(size_t size);
void* CudaRawMallocHost(size_t size);
void* CudaRawMallocManaged(size_t size);
void* CudaRawMallocHostZero(size_t size);
void* CudaRawHostGetDevicePtr(void* p);
void  CudaFree(void* t);
void  CudaFreeHost(void* t);
void  CudaRawCopyDeviceToHost(void* dest, const void* src, size_t count);
void  CudaRawCopyHostToDevice(void* dest, const void* src, size_t count);
void  CudaRawCopyDeviceToDevice(void* dest, const void* src, size_t count);
void  CudaSynchronizeDefaultStream();
void  CudaRawCopyToSymbol(const void* dest, const void* src, size_t count);

void CudaHostRegister(void* ptr, size_t size);
void CudaHostUnregister(void* ptr);

/// \returns a vector device properties of all GPU devices on the machine. The index
/// corresponds to the original device id.  If there is a problem with a device,
/// the cudaDeviceProp::uuid field will be set to all zeros.
struct CudaDeviceProperties 
{
    struct cudaDeviceProp deviceProperties;
    std::string errorMessage;
};

std::vector<CudaDeviceProperties> CudaAllGpuDevices();

// Manually check if an error has occured.  Will capture
// asynchronous errors that have not yet happened since
// the last CUDA API call.  This also might be the only
// way to catch cuda kernel launch errors.
void ThrowIfCudaError();

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
void CudaCopyDeviceToHost(T* dest, const T* src, size_t count) { CudaRawCopyDeviceToHost(dest, src, count*sizeof(T)); }
template <typename T>
void CudaCopyHostToDevice(T* dest, const T* src, size_t count) { CudaRawCopyHostToDevice(dest, src, count*sizeof(T)); }
template <typename T>
void CudaCopyDeviceToDevice(T* dest, const T* src, size_t count) { CudaRawCopyDeviceToDevice(dest, src, count*sizeof(T)); }
template <typename T>
void CudaCopyToSymbol(const T* dest, const T* src) { CudaRawCopyToSymbol(dest, src, sizeof(T)); }

}} // ::Pacbio::Cuda

#endif //PACBIO_CUDA_PB_CUDA_RUNTIME_H
