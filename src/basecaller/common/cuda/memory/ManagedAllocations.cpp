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

#include <pacbio/dev/profile/ScopedProfilerChain.h>

SMART_ENUM(ManagedAllocations, HostAllocate, GpuAllocate, HostDeallocate, GpuDeallocate);
using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<ManagedAllocations>;

namespace PacBio {
namespace Cuda {
namespace Memory {

namespace {

// Helper class to keep track of memory high water marks
class AllocStat
{
public:
    void AllocationSize(size_t size)
    {
        currentBytes_ += size;
        peakBytes_ = std::max(currentBytes_, peakBytes_);
    }
    void DeallocationSize(size_t size)
    {
        assert(size <- currentBytes_);
        currentBytes_ -= size;
    }

    size_t PeakMemUsage() const { return peakBytes_; }
private:
    size_t currentBytes_{0};
    size_t peakBytes_{0};
};

// Memory management class that allows re-use of memory allocations.
// Really only suitable for regular allocations of a small variety of
// sizes.  If you need frequent allocations of unpredictable size,
// this class will probably behave poorly and you should use a real
// general purpose allocator, which this class definitely is not!
//
// Note: This entire data structure is guarded by a single mutex,
//       which obviously is not scalable if you increase the number
//       of threads or the frequency of allocations.  With current
//       mongo usage patterns this really isn't an issue, as
//       individual batches take 10s of milli-seconds to complete,
//       and individual allocations only take 10s of micro-seconds
//       to be satisfied.  However if usage patterns change, this
//       may have to be migrated to something that either does more
//       fine grained locking, or is lock-free altogether.
struct AllocationManager
{
private:
    AllocationManager() = default;

    AllocationManager(const AllocationManager&) = delete;
    AllocationManager(AllocationManager&&) = delete;
    AllocationManager& operator=(const AllocationManager&) = delete;
    AllocationManager& operator=(AllocationManager&&) = delete;
public:

    // Singleton access function.  Static variables in a function have
    // a well defined and sane initialization guarantee, while actual
    // global variables do not (since the ordering of initialization
    // between different translation units is indeterminate)
    static AllocationManager& GetManager()
    {
        static AllocationManager manager;
        return manager;
    };

    ~AllocationManager()
    {
        if (enabled_)
        {
            PBLOG_ERROR << "Destroying AllocationManager while allocations are active.  "
                        << "This indicates not all memory was freed before static teardown "
                        << "which usually causes an error, as the cuda runtime may no longer "
                        << "exit.  We're probably about to die gracelessly and now you know why!";

            FlushPools();
        }
    }

    // After calling this, calls to the `Return` functions will always
    // store memory for future use, and host allocations will use
    // pinned memory.  Beware no memory is ever freed unless
    // you call `EnablePooling`.  For certain usage patterns
    // (e.g. irregular and unpredictable allocation sizes), use of
    // AllocationManager will effectively look like a memory leak
    void EnablePerformanceMode()
    {
        std::lock_guard<std::mutex> lm(m_);
        enabled_ = true;
        pinned_ = true;
    }

    // After calling this, calls to the `Return` functions
    // will immediately destroy all allocations it receives
    void DisablePerformanceMode()
    {
        std::lock_guard<std::mutex> lm(m_);
        enabled_ = false;
        pinned_ = false;
        FlushPoolsImpl();
    }

    SmartHostAllocation GetHostAlloc(size_t size, const AllocationMarker& marker)
    {
        SmartHostAllocation ptr;
        if (size == 0) return ptr;

        std::lock_guard<std::mutex> lm(m_);
        auto& queue = hostAllocs_[size];
        if (queue.size() > 0)
        {
            ptr = std::move(queue.front());
            queue.pop_front();
            ptr.AllocID(marker.AsHash());
        } else {
            ptr = SmartHostAllocation(size, pinned_, marker.AsHash());
        }

        assert(allocMarkers_.count(marker.AsHash()) == 0
               || allocMarkers_.at(marker.AsHash()) == marker);
        allocMarkers_.emplace(marker.AsHash(), marker);
        hostStats_[marker.AsHash()].AllocationSize(size);

        return ptr;
    }

    SmartDeviceAllocation GetDeviceAlloc(size_t size, const AllocationMarker& marker)
    {
        SmartDeviceAllocation ptr;
        if (size == 0) return ptr;

        std::lock_guard<std::mutex> lm(m_);
        auto& queue = devAllocs_[size];
        if (queue.size() > 0)
        {
            ptr = std::move(queue.front());
            queue.pop_front();
            ptr.AllocID(marker.AsHash());
        } else {
            ptr = SmartDeviceAllocation(size, marker.AsHash());
        }

        assert(allocMarkers_.count(marker.AsHash()) == 0
               || allocMarkers_.at(marker.AsHash()) == marker);
        allocMarkers_.emplace(marker.AsHash(), marker);
        devStats_[marker.AsHash()].AllocationSize(size);

        return ptr;
    }

    void ReturnHostAlloc(SmartHostAllocation alloc)
    {
        std::lock_guard<std::mutex> lm(m_);

        if (!enabled_ || alloc.Pinned() != pinned_) return;
        if (alloc.AllocID() != 0)
            hostStats_[alloc.AllocID()].DeallocationSize(alloc.size());
        hostAllocs_[alloc.size()].push_front(std::move(alloc));
    }

    void ReturnDevAlloc(SmartDeviceAllocation alloc)
    {
        std::lock_guard<std::mutex> lm(m_);

        if (!enabled_) return;
        devStats_[alloc.AllocID()].DeallocationSize(alloc.size());
        devAllocs_[alloc.size()].push_front(std::move(alloc));
    }

    void FlushPools()
    {
        std::lock_guard<std::mutex> lm(m_);

        FlushPoolsImpl();
    }

private:
    // Clears out all allocation pools (and generates a usage report)
    void FlushPoolsImpl()
    {
        // Generate reports, to see what sections of code are allocating
        // the most memory
        auto Report = [&](auto& stats) {

            std::vector<std::pair<std::string, size_t>> highWaters;
            for (auto& kv : stats)
            {
                const auto& name = allocMarkers_.at(kv.first).AsString();
                const auto value = kv.second.PeakMemUsage();
                highWaters.emplace_back(std::make_pair(name, value));
            }

            std::sort(highWaters.begin(),
                      highWaters.end(),
                      [](const auto& l, const auto& r) {return l.second > r.second; });
            size_t sum = 0;
            sum = std::accumulate(highWaters.begin(),
                                  highWaters.end(),
                                  sum,
                                  [](const auto& a, const auto& b) { return a+b.second; });

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
        Report(devStats_);
        PBLOG_INFO << "------------------------------------------";
        PBLOG_INFO << "----Host Memory high water mark report----";
        Report(hostStats_);
        PBLOG_INFO << "------------------------------------------";
        PBLOG_INFO << "Note: Different filter stages consume memory at different times.  "
                   << "The sum of high water marks for individual high watermarks won't "
                   << "be quite the same as the high watermark for the application as a whole";

        // Delete all allocations (and statistics) we're currently holding on to.
        hostAllocs_.clear();
        devAllocs_.clear();
        hostStats_.clear();
        devStats_.clear();
        allocMarkers_.clear();
    }


    std::map<size_t, std::deque<SmartHostAllocation>> hostAllocs_;
    std::map<size_t, std::deque<SmartDeviceAllocation>> devAllocs_;

    std::map<size_t, AllocStat> hostStats_;
    std::map<size_t, AllocStat> devStats_;

    std::map<size_t, AllocationMarker> allocMarkers_;

    std::mutex m_;
    bool enabled_ = false;
    bool pinned_ = false;
};

}

SmartHostAllocation GetManagedHostAllocation(size_t size, const AllocationMarker& marker)
{
    static thread_local size_t counter = 0;
    counter++;
    auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
    Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
    auto tmp = prof.CreateScopedProfiler(ManagedAllocations::HostAllocate);
    return AllocationManager::GetManager().GetHostAlloc(size, marker);
}

SmartDeviceAllocation GetManagedDeviceAllocation(size_t size, const AllocationMarker& marker)
{
    static thread_local size_t counter = 0;
    counter++;
    auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
    Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
    auto tmp = prof.CreateScopedProfiler(ManagedAllocations::GpuAllocate);
    return AllocationManager::GetManager().GetDeviceAlloc(size, marker);
}

void ReturnManagedHostAllocation(SmartHostAllocation alloc)
{
    if (alloc.size() == 0) return;
    static thread_local size_t counter = 0;
    counter++;
    auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
    Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
    auto tmp = prof.CreateScopedProfiler(ManagedAllocations::HostDeallocate);
    AllocationManager::GetManager().ReturnHostAlloc(std::move(alloc));
}

void ReturnManagedDeviceAllocation(SmartDeviceAllocation alloc)
{
    if (alloc.size() == 0) return;
    static thread_local size_t counter = 0;
    counter++;
    auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
    Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
    auto tmp = prof.CreateScopedProfiler(ManagedAllocations::GpuDeallocate);
    AllocationManager::GetManager().ReturnDevAlloc(std::move(alloc));
}

void EnablePerformanceMode()
{
    AllocationManager::GetManager().EnablePerformanceMode();
}

void DisablePerformanceMode()
{
    // I don't think this report has much value for normal runs,
    // but may be useful if profiling or diagnosing performance
    // issues, in which case we're probably using RelWithDebInfo
    // and this will turn on.
#ifndef NDEBUG
    Profiler::FinalReport();
#endif
    AllocationManager::GetManager().DisablePerformanceMode();
}

}}} // ::PacBio::Cuda::Memory
