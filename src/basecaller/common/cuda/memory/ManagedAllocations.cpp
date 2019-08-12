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

#include <algorithm>
#include <map>
#include <mutex>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <tuple>

#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/logging/Logger.h>

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

    auto begin()
    {
        std::lock_guard<std::mutex> l(m_);
        return data_.begin();
    }

    auto end()
    {
        std::lock_guard<std::mutex> l(m_);
        return data_.end();
    }
private:
    std::map<Key, Val> data_;
    std::mutex m_;
};

class AtomicString
{
public:
    AtomicString(const std::string& dat = "")
        : dat_(dat)
    {}

    AtomicString& operator=(const std::string& s)
    {
        std::lock_guard<std::mutex> l(m_);
        dat_ = s;
        return *this;
    }

    operator std::string() const
    {
        std::lock_guard<std::mutex> l(m_);
        std::string ret = dat_;
        return ret;
    }
private:
    std::string dat_;
    mutable std::mutex m_;
};

class AllocStat
{
public:
    void Take(size_t size)
    {
        std::lock_guard<std::mutex> l(m_);
        current += size;
        high = std::max(current.load(), high.load());
    }
    void Return(size_t size)
    {
        std::lock_guard<std::mutex> l(m_);
        current -= size;
    }

    size_t High() const { return high; }
private:
    std::atomic<size_t> current{0};
    std::atomic<size_t> high{0};
    std::mutex m_;
};

template <typename T>
struct Locker
{
    template <typename U>
    friend class AccessLock;

    using Payload = T;

    void Enable() { enabled_ = true; }
private:

    operator T&() { return data_; }
    T data_;
    std::atomic<size_t> activeThreads_{0};
    std::atomic<bool> enabled_{false};
};

template <typename T>
class AccessLock
{
public:
    enum Mode
    {
        Weak,
        Strong
    };
    AccessLock(Locker<T>& lock, Mode mode)
        : lock_(lock)
        , payload_(lock)
        , mode_(mode)
    {
        if (mode == Weak)
        {
            if (lock.enabled_)
            {
                lock.activeThreads_++;
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
                if (!lock.enabled_)
                {
                    lock.activeThreads_--;
                } else {
                    granted_ = true;
                }
            }
        } else {
            lock.enabled_ = false;
            while (lock.activeThreads_ > 0) {}
            granted_ = true;
        }
    }

    operator bool() { return granted_; }

    ~AccessLock()
    {
        if (mode_ == Weak)
          lock_.activeThreads_--;
    }

    T* operator->() { assert(granted_); return &payload_; }

private:

    Locker<T>& lock_;
    T& payload_;
    bool granted_ = false;
    Mode mode_;
};


struct Manager
{
    SmartHostAllocation GetHostAlloc(size_t size, bool pinned, const AllocationMarker& marker)
    {
        bool success = false;
        SmartHostAllocation ptr;
        if (size == 0) return ptr;
        if (auto lock = AccessLock<Maps>(locker_, AccessLock<Maps>::Weak))
        {
            auto& queue = lock->hostAllocs_[size];
            success = queue.TryPop(ptr);

#ifndef NDEBUG
            std::string loc = lock->allocMarkers_[marker.AsHash()];
            // Nothing *really* breaks if we actually have a hash collision, except
            // that the reports will lie to you (to some extent).  If this assert
            // presents a burdon it can be re-evaluated
            assert(loc == "" || loc == marker.AsString());
#endif
            lock->allocMarkers_[marker.AsHash()] = marker.AsString();
            lock->hostStats_[marker.AsHash()].Take(size);
        }
        if (success)
            ptr.Hash(marker.AsHash());
        else
            ptr = SmartHostAllocation(size, pinned, marker.AsHash());

        return ptr;
    }

    SmartDeviceAllocation GetDeviceAlloc(size_t size, const AllocationMarker& marker)
    {
        bool success = false;
        SmartDeviceAllocation ptr;
        if (size == 0) return ptr;
        if (auto lock = AccessLock<Maps>(locker_, AccessLock<Maps>::Weak))
        {
            auto& queue = lock->devAllocs_[size];
            success = queue.TryPop(ptr);

#ifndef NDEBUG
            std::string loc = lock->allocMarkers_[marker.AsHash()];
            // Nothing *really* breaks if we actually have a hash collision, except
            // that the reports will lie to you (to some extent).  If this assert
            // presents a burden it can be re-evaluated
            assert(loc == "" || loc == marker.AsString());
#endif
            lock->allocMarkers_[marker.AsHash()] = marker.AsString();
            lock->devStats_[marker.AsHash()].Take(size);
        }
        if (success)
            ptr.Hash(marker.AsHash());
        else
            ptr = SmartDeviceAllocation(size, marker.AsHash());


        return ptr;
    }

    void ReturnHost(SmartHostAllocation alloc)
    {
        if (auto lock = AccessLock<Maps>(locker_, AccessLock<Maps>::Weak))
        {
            lock->hostStats_[alloc.Hash()].Return(alloc.size());
            lock->hostAllocs_[alloc.size()].Push(std::move(alloc));
        }
    }

    void ReturnDev(SmartDeviceAllocation alloc)
    {
        if (auto lock = AccessLock<Maps>(locker_, AccessLock<Maps>::Weak))
        {
            lock->devStats_[alloc.Hash()].Return(alloc.size());
            lock->devAllocs_[alloc.size()].Push(std::move(alloc));
        }
    }

    void Enable() { locker_.Enable(); }
    void Disable()
    {
        auto lock = AccessLock<Maps>(locker_, AccessLock<Maps>::Strong);
        assert(lock);

        auto Report = [&](auto& stats) {
            std::vector<std::pair<std::string, size_t>> highWaters;
            for (auto& kv : stats)
            {
                const auto& name = std::string(lock->allocMarkers_[kv.first]);
                const auto value = kv.second.High();
                highWaters.emplace_back(std::make_pair(name, value));
            }

            std::sort(highWaters.begin(),
                      highWaters.end(),
                      [&](const auto& l, const auto& r) {return l.second > r.second; });
            size_t sum = 0;
            sum = std::accumulate(highWaters.begin(),
                                  highWaters.end(),
                                  sum,
                                  [&](const auto& a, const auto& b) { return a+b.second; });

            std::stringstream msg;
            const float mb = static_cast<float>(1<<20);
            msg << "High water marks sum: " << sum / mb << " MB\n";
            for (const auto& val : highWaters)
            {
                std::ios state(nullptr);
                state.copyfmt(msg);
                msg << "\t" << std::setprecision(2) << std::setw(5) << 100.0f * val.second / sum << "% : ";
                msg.copyfmt(state);
                msg << val.first << ": ";
                msg << val.second / mb << " MB\n";
            }
            Logging::LogStream ls(Logging::LogLevel::INFO);
            ls << msg.str();
        };

        PBLOG_INFO << "----GPU Memory high water mark report-----";
        Report(lock->devStats_);
        PBLOG_INFO << "------------------------------------------";
        PBLOG_INFO << "----Host Memory high water mark report----";
        Report(lock->hostStats_);
        PBLOG_INFO << "------------------------------------------";
        PBLOG_INFO << "Note: Different filter stages consume memory at different times.  "
                   << "The sum of high water marks for individual high watermarks won't "
                   << "be quite the same as the high watermark for the application as a whole";

        lock->hostAllocs_.clear();
        lock->devAllocs_.clear();
        lock->hostStats_.clear();
        lock->devStats_.clear();
        lock->allocMarkers_.clear();
    }

    struct Maps
    {
        ThreadSafeMap<size_t, ThreadSafeQueue<SmartHostAllocation>> hostAllocs_;
        ThreadSafeMap<size_t, ThreadSafeQueue<SmartDeviceAllocation>> devAllocs_;

        ThreadSafeMap<size_t, AllocStat> hostStats_;
        ThreadSafeMap<size_t, AllocStat> devStats_;

        ThreadSafeMap<size_t, AtomicString> allocMarkers_;
    };

    Locker<Maps> locker_;

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
    if (alloc.size() == 0) return;
    GetManager().ReturnHost(std::move(alloc));
}

void ReturnManagedDeviceAllocation(SmartDeviceAllocation alloc)
{
    if (alloc.size() == 0) return;
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
