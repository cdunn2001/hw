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

#include "ManagedAllocations.h"

#include <map>
#include <mutex>

#include <pacbio/ipc/ThreadSafeQueue.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

namespace {

// This is not meant to be a very robust structure, and probably shouldn't be made
// more public than this file.  In particular only the structure of the map itself
// is thread safe.  Mutating references of the contained type is *not* thread safe
// unless Val is itself a thread safe type.
template <typename Key, typename Val>
struct ThreadSafeMap
{
    Val& operator[] (const Key& key)
    {
        std::lock_guard<std::mutex> l(m_);
        return data_[key];
    }
    void clear()
    {
        std::lock_guard<std::mutex> l(m_);
        data_.clear();
    }
private:
    std::map<Key, Val> data_;
    std::mutex m_;
};

struct AllocStat
{
    std::atomic<size_t> current{0};
    std::atomic<size_t> high{0};
    std::atomic<size_t> checkedOut{0};
};

struct Manager
{
    SmartHostAllocation GetHostAlloc(size_t size, bool pinned, const AllocationMarker& marker)
    {
        bool done = false;
        SmartHostAllocation ptr;
        if (enabled_)
        {
            activeThreads++;
            auto& queue = hostAllocs_[size];
            done = queue.TryPop(ptr);
            activeThreads--;
        }
        if (done) return ptr;
        else return SmartHostAllocation(size);
    }

    SmartDeviceAllocation GetDeviceAlloc(size_t size, const AllocationMarker& marker)
    {
        bool done = false;
        SmartDeviceAllocation ptr;
        if (enabled_)
        {
            activeThreads++;
            auto& queue = devAllocs_[size];
            done = queue.TryPop(ptr);
            activeThreads--;
        }
        if (done) return ptr;
        else return SmartDeviceAllocation(size);
    }

    void ReturnHost(SmartHostAllocation alloc)
    {
        if (enabled_)
        {
            activeThreads++;
            auto& queue = hostAllocs_[alloc.size()];
            queue.Push(std::move(alloc));
            activeThreads--;
        }
    }

    void ReturnDev(SmartDeviceAllocation alloc)
    {
        if (enabled_)
        {
            activeThreads++;
            auto& queue = devAllocs_[alloc.size()];
            queue.Push(std::move(alloc));
            activeThreads--;
        }
    }

    void Enable() { enabled_ = true; }
    void Disable()
    {
        enabled_ = false;
        while(activeThreads > 0) {}
        hostAllocs_.clear();
        devAllocs_.clear();
        hostStats_.clear();
        devStats_.clear();
    }

    ThreadSafeMap<size_t, ThreadSafeQueue<SmartHostAllocation>> hostAllocs_;
    ThreadSafeMap<size_t, ThreadSafeQueue<SmartDeviceAllocation>> devAllocs_;

    ThreadSafeMap<size_t, AllocStat> hostStats_;
    ThreadSafeMap<size_t, AllocStat> devStats_;

    std::atomic<size_t> activeThreads{0};
    std::atomic<bool> enabled_{false};
};

Manager& GetManager()
{
    static Manager manager;
    return manager;
};


}

SmartHostAllocation GetManagedHostAllocation(size_t size, bool pinned, const AllocationMarker& marker)
{
    return GetManager().GetHostAlloc(size, pinned, marker);
}

SmartDeviceAllocation GetManagedDeviceAllocation(size_t size, const AllocationMarker& marker, bool throttle)
{
    return GetManager().GetDeviceAlloc(size, marker);
}

void ReturnManagedHostAllocation(SmartHostAllocation alloc)
{
    GetManager().ReturnHost(std::move(alloc));
}

void ReturnManagedDeviceAllocation(SmartDeviceAllocation alloc)
{
    GetManager().ReturnDev(std::move(alloc));
}

void EnablePooling()
{
    GetManager().Enable();
}

void DisablePooling()
{
    GetManager().Disable();
}

}}} // ::PacBio::Cuda::Memory
