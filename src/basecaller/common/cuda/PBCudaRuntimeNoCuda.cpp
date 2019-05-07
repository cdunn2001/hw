
// Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
//  Alternative definitions of the basic functions declared in PBCudaRuntime.h
//  that have no CUDA dependency.

#include "PBCudaRuntime.h"

#include <cstdlib>
#include <pacbio/PBException.h>

//#include <thread>

using std::malloc;

namespace PacBio {
namespace Cuda {

namespace {

template <typename T>
void cudaCheckErrors(T&& result)
{
//    if (result != cudaSuccess)
//    {
//        std::stringstream ss;
//        ss << "Cuda failure, error code: " << cudaGetErrorName(result);
//        throw PBException(ss.str());
//    }
}

}   // anonymous namespace

void* CudaRawMalloc(size_t size)
{ return malloc(size); }

void* CudaRawMallocHost(size_t size)
{ return malloc(size); }

void* CudaRawMallocManaged(size_t size)
{ return malloc(size); }

void* CudaRawMallocHostZero(size_t size)
{ return malloc(size); }


void* CudaRawHostGetDevicePtr(void* p)
{
    // FIXME: What should this do?
    return nullptr;
}


void CudaFree(void* t)
{ free(t); }

void CudaFreeHost(void* t)
{ free(t); }


void CudaSynchronizeDefaultStream()
{
    // FIXME: What should this do?
}

void CudaRawCopyHost(void* dest, void* src, size_t size)
{
    // FIXME: What should this do?
}

void CudaRawCopyDevice(void* dest, void* src, size_t size)
{
    // FIXME: What should this do?
}

}}  // namespace PacBio::Cuda
