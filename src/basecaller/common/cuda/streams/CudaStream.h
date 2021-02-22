//  Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_CUDA_CUDA_STREAM_H
#define PACBIO_CUDA_CUDA_STREAM_H

#include <common/cuda/PBCudaRuntime.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>
#include <pacbio/utilities/Finally.h>

namespace PacBio {
namespace Cuda {

class CudaStream
{
public:
    explicit CudaStream(int priority)
    {
        stream_ = CreateStream(priority);
    }

    Utilities::Finally SetAsDefaultStream() const
    {
        if (threadStream_ != cudaStreamPerThread)
            throw PBException("Setting CudaStream as the thread default "
                              "stream when another instance already is!");

        CudaSynchronizeDefaultStream();
        threadStream_ = stream_;
        return Utilities::Finally([](){
            CudaSynchronizeDefaultStream();
            threadStream_ = cudaStreamPerThread;
        });
    }

    // No copy/move for simplicity.  Note that
    // cudaStream_t is merely a typedef over
    // a pointer.  A copy doesn't make much sense
    // while preserving RAII semantics, and if you
    // want to enable move semantics, be sure to
    // null out the moved-from "pointer", though in
    // this case be aware that the "null pointer" in
    // this context also specificall overloaded to
    // mean "cudaStreamPerThread"
    CudaStream(const CudaStream&) = delete;
    CudaStream(CudaStream&& o) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream& operator=(CudaStream&& o) = delete;

    ~CudaStream()
    {
        if (stream_ && threadStream_ == stream_)
            PBLOG_ERROR << "Destroying a cuda stream that is actively"
                        << " being used as the per-thread default!";
        if (stream_) Cuda::DestroyStream(stream_);
    }

    operator cudaStream_t()
    {
        return stream_;
    }

    static cudaStream_t ThreadStream()
    {
        return threadStream_;
    }

private:
    // raw cuda stream.  This is just a typedef over a pointer
    // so treat it as such.
    cudaStream_t stream_;

    static thread_local cudaStream_t threadStream_;
};

}} // ::PacBio::Cuda

#endif // PACBIO_CUDA_CUDA_STREAM_H
