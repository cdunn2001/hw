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

#ifndef PACBIO_CUDA_STREAM_MONITORS_H
#define PACBIO_CUDA_STREAM_MONITORS_H

#include <unordered_map>

#include <common/cuda/streams/KernelLaunchInfo.h>
#include <common/cuda/streams/CudaEvent.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>

// This file provides classes for monitoring streams of gpu execution.  The
// intent is that a class that controls the lifetime of resources potentially
// in use on the GPU would posses an instance of a stream monitor.  With each
// kernel invokation it can use a KernelLaunchInfo instance to call the
// Update function, which can then perform various sanity checks.  If the class
// detects any data races, or is destroyed before all associated kernels are
// completed, it will raise an error.

namespace PacBio {
namespace Cuda {

// Many errors happen during the destruction of
// an object, so throwing is not really an option.
// These functions provide provide a primitive API
// for keeping track of these errors so that
// interested code can later detect (and possibly
// throw) if something went wrong.
void AddStreamError();
void ResetStreamErrors();
size_t StreamErrorCount();

// Class used to monitor and control execution for an object designed to be used
// only on a *single* cuda stream at any one time.  The object can be used on
// different streams, but only if assocated kernels on one stream have all
// completed before any work on a separate stream commences.
class SingleStreamMonitor
{
public:
    SingleStreamMonitor()
        : lastThread_{KernelLaunchInfo::NoThreadId} // reserve tid meaning "no-one"
    {}

    SingleStreamMonitor(const SingleStreamMonitor&) = delete;
    SingleStreamMonitor(SingleStreamMonitor&& o) = default;
    SingleStreamMonitor& operator=(const SingleStreamMonitor&) = delete;
    SingleStreamMonitor& operator=(SingleStreamMonitor&& o) = default;

    // Records the last seen cuda event.  It will throw if we are switching
    // streams while we are still being used on the previous stream
    void Update(const KernelLaunchInfo& info)
    {
        // It's an error to use this type on two different streams concurrently
        // If someone merely forgot to add a synchronization point we will add
        // one now to be safe, but this may also indicate a wrong argument has
        // accidentally passed to the kernel.
        if (lastThread_ != KernelLaunchInfo::NoThreadId && info.ThreadId() != lastThread_)
        {
            if (!latestEvent_->IsCompleted())
            {
                PBLOG_WARN << "Unexpected launch of cuda kernel with data already in use in another stream.  "
                           << "Automatically synchronizing to try and maintain correctness, but this may "
                           << "indicate a developer bug.";
                latestEvent_->WaitForCompletion();
                AddStreamError();
            }
        }

        // if we're on the same thread/stream, we can just forget the old event and take the new
        // latest one, as operations within the same stream execute in order
        latestEvent_ = info.Event();
        lastThread_ = info.ThreadId();
    }

    ~SingleStreamMonitor()
    {
        Reset();
    }

    // Makes sure that all outstanding work is completed
    void Reset()
    {
        if (latestEvent_)
        {
            if (!latestEvent_->IsCompleted())
            {
                PBLOG_WARN << "Unexpectedly trying to reset stream monitor while currently in use "
                           << "on the gpu.  We'll automatically synchronize to avoid segmentation "
                           << "violations, but this may indicate a developer bug.";
                latestEvent_->WaitForCompletion();
                AddStreamError();
            }
        }

        lastThread_ = KernelLaunchInfo::NoThreadId;
        latestEvent_ = nullptr;
    }
private:
    uint32_t lastThread_;
    std::shared_ptr<const CudaEvent> latestEvent_;
};

// Class used to monitor and control execution for an object designed to be used
// only on multiple cuda streams.  Really only suitable for containers of const
// data, else you'll likely have different kernels mutating each others data
// in surprising fashions.  The main constraint in this class is to make sure
// *all* kernels associated with this on *all* streams have completed before
// we can destroy ourselves.
class MultiStreamMonitor
{
public:
    MultiStreamMonitor() = default;

    MultiStreamMonitor(const MultiStreamMonitor&) = delete;
    MultiStreamMonitor(MultiStreamMonitor&&) = default;
    MultiStreamMonitor& operator=(const MultiStreamMonitor&) = delete;
    MultiStreamMonitor& operator=(MultiStreamMonitor&&) = default;

    // Any number of threads/streams are allowed to use this type
    // concurrently.  We're just going to keep track of the latest
    // operation in each thread that comes through.
    void Update(const KernelLaunchInfo& info)
    {
        eventMap_[info.ThreadId()] = info.Event();
    }

    ~MultiStreamMonitor()
    {
        for (auto& kv : eventMap_)
        {
            if (kv.second && !kv.second->IsCompleted())
            {
                PBLOG_WARN << "Unexpectedly trying to destroy data while currently in use "
                           << "on the gpu.  We'll automatically synchronize to avoid segmentation "
                           << "violations, but this may indicate a developer bug.";
                kv.second->WaitForCompletion();
                AddStreamError();
            }
        }
    }
private:
    std::unordered_map<uint32_t, std::shared_ptr<const CudaEvent>> eventMap_;
};

}} // ::PacBio::Cuda

#endif // PACBIO_CUDA_STREAM_MONITORS_H
