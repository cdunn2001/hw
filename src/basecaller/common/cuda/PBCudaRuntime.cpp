//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

#include "PBCudaRuntime.h"
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

void ThrowIfCudaError()
{
    cudaCheckErrors(cudaGetLastError());
}

size_t RequiredRegisterCount(const void* func)
{
    cudaFuncAttributes funcAttrib;
    cudaCheckErrors(cudaFuncGetAttributes(&funcAttrib, func));
    return funcAttrib.numRegs;
}

size_t AvailableRegistersPerBlock()
{
    cudaDeviceProp props;
    cudaCheckErrors(cudaGetDeviceProperties(&props, 0));
    return props.regsPerBlock;
}

cudaEvent_t InitializeEvent()
{
    cudaEvent_t ret;
    cudaCheckErrors(::cudaEventCreate(&ret));
    return ret;
}

void DestroyEvent(cudaEvent_t event)
{
    cudaCheckErrors(::cudaEventDestroy(event));
}

void RecordEvent(cudaEvent_t event)
{
    cudaCheckErrors(::cudaEventRecord(event, cudaStreamPerThread));
}

void SyncEvent(cudaEvent_t event)
{
    cudaCheckErrors(::cudaEventSynchronize(event));
}

bool CompletedEvent(cudaEvent_t event)
{
    auto status = cudaEventQuery(event);

    // We need to specifically test for the "not ready" error
    // state as that is an expected potential result.  Once
    // that is out of the way do a normal error checking call
    // (to capture async errors), and if that everything comes
    // up clean then we know the event is formally completed as well.
    if (status == cudaErrorNotReady) return false;
    cudaCheckErrors(status);
    return true;
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

void CudaRawCopyHost(void* dest, void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost));
}

void CudaRawCopyDevice(void* dest, void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToDevice, cudaStreamPerThread));
}

void CudaRawCopyToSymbol(const void* dest, void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyToSymbol(dest, src, count));
}

}}
