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

#ifndef PACBIO_CUDA_KERNEL_LAUNCH_INFO_H
#define PACBIO_CUDA_KERNEL_LAUNCH_INFO_H

#include <cassert>
#include <cstdint>
#include <memory>

#include <common/cuda/streams/CudaEvent.h>

namespace PacBio {
namespace Cuda {

// This class records information about a kernel
// launch, namely the associated thread/stream
// and CudaEvent.  Only intended to be constructible
// by the kernel launching wrapper.  Any data types
// that wish/need to track kernels for which they were
// used should provide an overload for KernelArgConvert,
// which can be used to forward KernelLaunchInfo to
// that class.
class KernelLaunchInfo
{

public:
    // Special reserved tid to indicate no associated thread
    static constexpr uint32_t NoThreadId = 0;

private:
    template <typename FT, typename... LaunchParams>
    friend class LaunchManager;

    KernelLaunchInfo(std::shared_ptr<CudaEvent> event)
        : threadId_{NoThreadId} // temporary value
        , event_{std::move(event)}
    {
        // Keep track of how many unique threads come through this constructor,
        // and give each thread a unique identifier.
        static std::atomic<uint32_t> activeThreadCount{0};
        // initialization of a static thread_local only happens the first time
        // a thread passes through.  So atomically increment the activeThreadCount,
        // and save the result as our personal and unique id.
        static const thread_local uint32_t threadId = ++activeThreadCount;

        threadId_ = threadId;
        assert(threadId_ != NoThreadId);
    }

public:
    std::shared_ptr<const CudaEvent> Event() const { return event_; }
    uint32_t ThreadId() const { return threadId_; }
private:

    // Keeps track of which host thread / cuda stream used to launch
    // the associated kernel
    uint32_t threadId_;
    // Event used to track when/if the associated kernel has
    // completed
    std::shared_ptr<CudaEvent> event_;
};


}} // ::PacBio::Cuda

#endif // PACBIO_CUDA_KERNEL_LAUNCH_INFO_H
