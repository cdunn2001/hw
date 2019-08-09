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
        if (auto lock = AccessLock(accessState_, AccessLock::Weak))
        {
            auto& queue = hostAllocs_[size];
            done = queue.TryPop(ptr);
        }
        if (done)
        {
            ptr.Hash(marker.AsHash());
            return ptr;
        }
        else return SmartHostAllocation(size, pinned, marker.AsHash());
    }

    SmartDeviceAllocation GetDeviceAlloc(size_t size, const AllocationMarker& marker)
    {
        bool done = false;
        SmartDeviceAllocation ptr;
        if (auto lock = AccessLock(accessState_, AccessLock::Weak))
        {
            auto& queue = devAllocs_[size];
            done = queue.TryPop(ptr);
        }
        if (done)
        {
            ptr.Hash(marker.AsHash());
            return ptr;
        }
        else return SmartDeviceAllocation(size, marker.AsHash());
    }

    void ReturnHost(SmartHostAllocation alloc)
    {
        if (auto lock = AccessLock(accessState_, AccessLock::Weak))
        {
            auto& queue = hostAllocs_[alloc.size()];
            queue.Push(std::move(alloc));
        }
    }

    void ReturnDev(SmartDeviceAllocation alloc)
    {
        if (auto lock = AccessLock(accessState_, AccessLock::Weak))
        {
            auto& queue = devAllocs_[alloc.size()];
            queue.Push(std::move(alloc));
        }
    }

    void Enable() { accessState_.enabled_ = true; }
    void Disable()
    {
        auto lock = AccessLock(accessState_, AccessLock::Strong);
        assert(lock);

        hostAllocs_.clear();
        devAllocs_.clear();
        hostStats_.clear();
        devStats_.clear();
    }

    struct AccessState
    {
        std::atomic<size_t> activeThreads_{0};
        std::atomic<bool> enabled_{false};
    } accessState_;

    class AccessLock
    {
    public:
        enum Mode
        {
            Weak,
            Strong
        };
        AccessLock(AccessState& state, Mode mode)
            : state_(state)
            , mode_(mode)
        {
            if (mode == Weak)
            {
                if (state_.enabled_)
                {
                    state_.activeThreads_++;
                    // Another thread may have toggled enabled_ after we checked it
                    // but before we published our activeThreads increment. If it
                    // now reads as disabled, remove the effects of our increment.
                    // The other thread will either have seen it be incremented and
                    // is waiting for it to decrement, or it hasn't and is currently
                    // we need decrement and get out of here so as to not clobber any
                    // data.
                    //
                    // However if we get to this point and enabled is still true, then
                    // we are guaranteed that when any other thread toggles enabled off
                    // then it will have already seen our activeThreads update, and will
                    // wait for that to return to zero before doing anything
                    if (!state_.enabled_)
                    {
                        state_.activeThreads_--;
                    } else {
                        granted_ = true;
                    }
                }
            } else {
                state_.enabled_ = false;
                while (state_.activeThreads_ > 0) {}
                granted_ = true;
            }
        }

        operator bool() { return granted_; }

        ~AccessLock()
        {
            if (mode_ == Weak)
              state_.activeThreads_--;
        }

    private:

        AccessState& state_;
        bool granted_ = false;
        Mode mode_;
    };

    ThreadSafeMap<size_t, ThreadSafeQueue<SmartHostAllocation>> hostAllocs_;
    ThreadSafeMap<size_t, ThreadSafeQueue<SmartDeviceAllocation>> devAllocs_;

    ThreadSafeMap<size_t, AllocStat> hostStats_;
    ThreadSafeMap<size_t, AllocStat> devStats_;

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
