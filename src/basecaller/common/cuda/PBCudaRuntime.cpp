#include <pacbio/cuda/PBCudaRuntime.h>
#include <pacbio/PBException.h>

#include <cuda_runtime.h>

#include <thread>

namespace PacBio {
namespace Cuda {

namespace {

template <typename T>
void cudaCheckErrors(T&& result)
{
    if (result != cudaSuccess)
    {
        std::stringstream ss;
        ss << "Cuda failure, error code: " << cudaGetErrorName(result);
        throw PBException(ss.str());
    }
}

}

void* CudaRawMalloc(size_t size)
{
    void* ret;
    cudaCheckErrors(::cudaMalloc(&ret, size));
    return ret;
}
void* CudaRawMallocHost(size_t size)
{
    void* ret;
    cudaCheckErrors(::cudaMallocHost(&ret, size));
    return ret;
}
void* CudaRawMallocManaged(size_t size)
{
    void* ret;
    cudaCheckErrors(::cudaMallocManaged(&ret, size));
    return ret;
}

void* CudaRawMallocHostZero(size_t size)
{
   void* ret;
   cudaCheckErrors(::cudaMallocHost(&ret, size, cudaHostAllocMapped));
   return ret;
}
void* CudaRawHostGetDevicePtr(void* p)
{
    void* ret;
    // TODO only handles a single device!
    cudaCheckErrors(::cudaHostGetDevicePointer(&ret, p, 0));
    return ret;
}

void CudaFree(void* t)
{
    cudaCheckErrors(::cudaFree(t));
}

void CudaFreeHost(void* t)
{
    cudaCheckErrors(::cudaFreeHost(t));
}

void CudaSynchronizeDefaultStream()
{
    cudaCheckErrors(::cudaStreamSynchronize(cudaStreamPerThread));
}

void CudaRawCopyHost(void* dest, void* src, size_t size)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
}

void CudaRawCopyDevice(void* dest, void* src, size_t size)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
}

}}
