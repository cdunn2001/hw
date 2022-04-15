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

#include <cuda_runtime.h>

#include <vector>

#include <common/cuda/streams/CudaStream.h>
#include <pacbio/PBException.h>

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
        if (result == cudaErrorMemoryAllocation)
        {
            throw PBExceptionCust(ss.str(), CudaMemException);
        }
        else
        {
            throw PBException(ss.str());
        }
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
    cudaCheckErrors(::cudaEventRecord(event, CudaStream::ThreadStream()));
}

void SyncEvent(cudaEvent_t event)
{
    cudaCheckErrors(::cudaEventSynchronize(event));
}

SupportedStreamPriorities StreamPriorityRange()
{
    SupportedStreamPriorities ret;
    cudaCheckErrors(::cudaDeviceGetStreamPriorityRange(&ret.leastPriority, &ret.greatestPriority));
    return ret;
}

cudaStream_t CreateStream(int priority)
{
    cudaStream_t ret;
    cudaCheckErrors(::cudaStreamCreateWithPriority(&ret, cudaStreamNonBlocking, priority));
    return ret;
}

void DestroyStream(cudaStream_t stream)
{
    cudaCheckErrors(::cudaStreamDestroy(stream));
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
    cudaCheckErrors(::cudaStreamSynchronize(CudaStream::ThreadStream()));
}

void CudaRawCopyDeviceToHost(void* dest, const void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToHost, CudaStream::ThreadStream()));
}

void CudaRawCopyHostToDevice(void* dest, const void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, count, cudaMemcpyHostToDevice, CudaStream::ThreadStream()));
}

void CudaRawCopyDeviceToDevice(void* dest, const void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyAsync(dest, src, count, cudaMemcpyDeviceToDevice, CudaStream::ThreadStream()));
}

void CudaRawCopyToSymbol(const void* dest, const void* src, size_t count)
{
    cudaCheckErrors(::cudaMemcpyToSymbol(dest, src, count));
}

void CudaHostRegister(void* ptr, size_t size)
{
    // Calls to cudaHostRegister are mysteriously failing with a cudaErrorMemoryAllocation
    // error code.  While this is documented as a valid return code for this function, I've
    // no idea what may be causing it.  We clearly already have the main allocation in hand,
    // so it's something internal to the cuda runtime bookkeeping that is choking.
    //
    // As a preliminary stopgap, we're adding a retry mechanism in case that makes a difference.
    // For now this is an experimental change, and since it's a complete shot in the dark
    // we want to know if it's actually mitigating the issue, or if we're getting lucky/unlucky
    // and just not triggering the root cause.  So, no matter what if the first call fails then
    // we are still going to throw an exception.  If the retry logic doesn't fix the issue then we'll
    // continue to throw the same (perhaps somewhat confusing) cudaErrorMemoryAllocation message.
    // If the retry succeeds then we'll throw a separate exception, since that way things will
    // get reported back to us, as opposed to vanilla log messages.  Assuming we see this succeed
    // in the wild at least once, we'll return and do the sensible thing only throwing if we
    // completely fail to register this memory address.

    try
    {
        cudaCheckErrors(::cudaHostRegister(ptr, size, cudaHostRegisterPortable));
    } catch (const CudaMemException& ex)
    {
        // Try again a few times, to see if that helps...
        for (int retriesLeft = 10; retriesLeft >= 0; --retriesLeft)
        {
            try
            {
                // Sleep for a bit, just in case there is a timing issue
                // We're in no rush here, so I'm being "generous" (on computer
                // timescales)
                std::this_thread::sleep_for(std::chrono::seconds{1});

                cudaCheckErrors(::cudaHostRegister(ptr, size, cudaHostRegisterPortable));
                // If we succeeded, then we exit the loop (and throw a brand new exception
                // for visibility reasons)
                break;
            } catch (const CudaMemException& ex)
            {
                if (retriesLeft == 0)
                {
                    // Retries didn't help...  Rethrow the CudaMemException, and
                    // things will look like they did before this rety logic was added
                    throw;
                }
            }
        }
        // Hooray, if we got here then the rety logic worked!  Now to throw an exception
        // so that someone files a bug report and lets us know...
        throw PBException("Difficulty encountered when trying to register allocation with the cuda runtime.  "
                          "This is a high priority issue, and triggering this particular exception "
                          "actually means we may have a path forward.  Please file a ticket ASAP,"
                          " or even reach out to Ben Byington directly");
    }
}

void CudaHostUnregister(void* ptr)
{
    cudaCheckErrors(::cudaHostUnregister(ptr));
}

std::vector<CudaDeviceProperties> CudaAllGpuDevices()
{
    std::vector<CudaDeviceProperties> devices;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) count = 0;

    for(int idevice=0;idevice<count;idevice++)
    {
        CudaDeviceProperties properties;
        cudaError_t result = cudaGetDeviceProperties(&properties.deviceProperties, idevice);

        if (result != cudaSuccess)
        {
            // something needs to be pushed to the devices vector because
            // the index of the vector elements corresponds to the idevice ordinal number.
            memset(&properties.deviceProperties.uuid,0,sizeof(properties.deviceProperties.uuid));
            properties.errorMessage = cudaGetErrorName(result);
        }
        devices.push_back(properties);
    }

    return devices;
}

}} // end of namespace

