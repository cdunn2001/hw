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

#ifndef PACBIO_CUDA_CUDA_EVENT_H
#define PACBIO_CUDA_CUDA_EVENT_H

#include <atomic>

#include <common/cuda/PBCudaRuntime.h>

namespace PacBio {
namespace Cuda {

// RAII style class to control the lifecycle of
// a cuda event.  A cuda event is meant to serve
// as a lightweight way to monitor the progress
// of asynchronous streams.  Any call to `RecordEvent`
// will place a marker (of sorts) in the current
// (per thread) stream, which will not complete until
// all previous items in that stream are also finished.
//
// A couple quirks that might be surprising to those
// unfamiliar with cuda events:
// 1. Multiple calls to `RecordEvent` are allowed,
//    though they effectively overwrite each other
//    and only the most recent matters (usually).
// 2. If you are synchronizing and waiting for an
//    event to complete, and another thread calls
//    `RecordEvent`, the synchronization should be
//     unaffected by the new event record. (i.e. it's
//     time spent blocking will not be increased)
class CudaEvent
{
public:
    CudaEvent()
        : completed_{true}
        , event_{InitializeEvent()}
    {}

    // No copy/move for simplicity.  Note that
    // cudaEvent_t is merely a typedef over
    // a pointer.  A copy doesn't make much sense
    // while preserving RAII semantics, and if you
    // want to enable move semantics, be sure to
    // null out the moved-from pointer.
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& o) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent& operator=(CudaEvent&& o) = delete;

    ~CudaEvent()
    {
        if (event_) Cuda::DestroyEvent(event_);
    }

    // Note there is technically a tiny race condition
    // where one thread could have finished recordering
    // an event but not yet updated `completed_`, so
    // another thread might return true from `IsCompleted`.
    // That said, it's only a tiny exacerbation on an
    // already present race conditino you'd have if different
    // threads were recording and checking events using the
    // underlying CUDA API.  Just don't do that.
    void RecordEvent()
    {
        Cuda::RecordEvent(event_);
        completed_ = false;
    }

    bool IsCompleted() const
    {
        if (completed_) return true;
        completed_ = Cuda::CompletedEvent(event_);
        return completed_;
    }

    void WaitForCompletion() const
    {
        if (completed_) return;
        Cuda::SyncEvent(event_);
        completed_ = true;
    }

private:
    // using this atomic_bool to "cache" results and avoid hammering
    // the CUDA API if a lot of different entities querry the same
    // event (which may happen once per function argument to each
    // kernel)
    mutable std::atomic_bool completed_;
    // raw cuda event.  This is just a typedef over a pointer
    // so treat it as such.
    cudaEvent_t event_;
};

}} // ::PacBio::Cuda

#endif // PACBIO_CUDA_CUDA_EVENT_H
