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

#include <pacbio/datasource/AllocationSlicer.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/HugePage.h>

#include <pacbio/dev/profile/ScopedProfilerChain.h>

SMART_ENUM(ManagedAllocations, HostAllocate, GpuAllocate, HostDeallocate, GpuDeallocate);
using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<ManagedAllocations>;

namespace PacBio {
namespace Cuda {
namespace Memory {

namespace {

using namespace PacBio::Memory;
using namespace PacBio::DataSource;

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

struct MallocAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags = 0;
    static constexpr const char* description = "Standard Host Memory";

    using AllocationT = SmartAllocation;
    static SmartAllocation CreateAllocation(size_t size, const AllocationMarker& marker)
    {
        return SmartAllocation(size,
                               &malloc,
                               &free,
                               AllocationID{marker.AsHash()},
                               AllocationType{static_cast<size_t>(AllocatorMode::MALLOC)});
    }
};

struct PinnedAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags = IAllocator::CUDA_MEMORY;
    static constexpr const char* description = "Pinned Host Memory";

    using AllocationT = SmartAllocation;
    static SmartAllocation CreateAllocation(size_t size, const AllocationMarker& marker)
    {
        return SmartAllocation(size,
                               &CudaRawMallocHost,
                               &CudaFreeHost,
                               AllocationID{marker.AsHash()},
                               AllocationType{static_cast<size_t>(AllocatorMode::CUDA)});
    }
};

struct HugePinnedAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags =
          IAllocator::CUDA_MEMORY
        | IAllocator::HUGEPAGES
        | IAllocator::ALIGN_64B;

    static constexpr const char* description = "Pinned HugePage Host Memory";

    using AllocationT = SmartAllocation;
    static SmartAllocation CreateAllocation(size_t size, const AllocationMarker& marker)
    {
        return SmartAllocation(size,
                               &Allocate,
                               &Deallocate,
                               AllocationID{marker.AsHash()},
                               AllocationType{static_cast<size_t>(AllocatorMode::HUGE_CUDA)});
    }

    HugePinnedAllocator()
    {
        Allocator();
    }

private:
    static void* Allocate(size_t size)
    {
        return Allocator().Allocate(size);
    }

    static void Deallocate(void* ptr)
    {
        Allocator().Deallocate(ptr);
    }

    // Helper class, to do the extra work required to unify huge
    // page allocations with the cuda runtime
    struct HugePinnedHelper
    {
        static void* Malloc(size_t size)
        {
            auto ptr = PacBio::HugePage::Malloc(size);
            // This function both page-locks the allocation,
            // and registers the address range with the cuda
            // runtime
            CudaHostRegister(ptr, size);
            return ptr;
        }

        static void Free(void* ptr)
        {
            // This function unlocks the page, and updates
            // the cuda runtime bookkeeping to reflect that
            CudaHostUnregister(ptr);
            PacBio::HugePage::Free(ptr);
        }
    };

    // Singleton access.  We're already using a static and application
    // wide allocation cache, so it makes sense to similarly treat
    // the AllocationSlicer that backs it all
    static AllocationSlicer<HugePinnedHelper, 64>& Allocator()
    {
        static AllocationSlicer<HugePinnedHelper, 64> alloc_(1<<30);
        return alloc_;
    }
};

struct GpuAllocator
{
    static constexpr const char* description = "Gpu Memory";
    using AllocationT = SmartDeviceAllocation;
    static SmartDeviceAllocation CreateAllocation(size_t size, const AllocationMarker& marker)
    {
        return SmartDeviceAllocation(size, marker.AsHash());
    }
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
template <typename Allocator>
struct AllocationManager
{
private:
    AllocationManager() = default;

    AllocationManager(const AllocationManager&) = delete;
    AllocationManager(AllocationManager&&) = delete;
    AllocationManager& operator=(const AllocationManager&) = delete;
    AllocationManager& operator=(AllocationManager&&) = delete;
public:

    using AllocationT = typename Allocator::AllocationT;

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
        if (cache_)
        {
            // Note: The use of a static AllocationManager for allocation caching
            //       opens us up to some potential error modes.  In particular,
            //       it's somewhere between difficult and impossible to control
            //       destruction order of objects with static storage duration.
            //       Technically it happens in reverse order of construction, but
            //       unless you are very careful to only use things in your destructor
            //       that are used in your constructor, odds are any globals used in
            //       the destructor of a static object are already dead.
            //
            //       This means that freeing any allocations after main exits is probably
            //       fatal.  The core problem is that the cuda runtime has possibly torn
            //       down, so any calls to that API are likely to fail hard.  Beyond that
            //       the HugePinnedAllocator has some static data that is also probably
            //       problematic, but not really worth wrestling with when the cuda problem
            //       can't be solved.
            //
            //       We could move away from a static/global allocation cache, which would
            //       fix most (but not all) of the issues.  However mongo started out with
            //       non-global caching, and it was cumbersom enough to use that we migrated
            //       to this static solution.  We can always revisit that design choice, but
            //       for now things work and these problems are mostly theoretical.
            //
            //       So at the end of the day, it's simply required that you make sure any
            //       allocations are freed before main exits, as well as all caching disabled.
            PBLOG_ERROR << "Destroying AllocationManager while cache is still active!  "
                        << "This indicates not all memory was freed before static teardown "
                        << "which usually causes an error, as global/static objects involved "
                        << "in deallocations, such as the cuda runtime itself, may already be "
                        << "dead.  We're probably about to die gracelessly and now you know why!";

            FlushPools();
        }
    }

    // After calling this, calls to the `Return` functions will always
    // store memory for future use, and host allocations will use
    // pinned memory.  Beware no memory is ever freed until
    // you call `DisableCaching`.  For certain usage patterns
    // (e.g. irregular and unpredictable allocation sizes), use of
    // AllocationManager will effectively look like a memory leak
    void EnableCaching()
    {
        std::lock_guard<std::mutex> lm(m_);
        cache_ = true;
    }

    // After calling this, calls to the `Return` functions
    // will immediately destroy all allocations it receives
    void DisableCaching()
    {
        std::lock_guard<std::mutex> lm(m_);
        cache_ = false;
        FlushPoolsImpl();
    }

    // Return type is either SmartAllocation or SmartDeviceAllocation
    auto GetAlloc(size_t size, const AllocationMarker& marker)
    {
        AllocationT ptr;
        if (size == 0) return ptr;

        std::lock_guard<std::mutex> lm(m_);
        auto& queue = allocs_[size];
        if (queue.size() > 0)
        {
            ptr = std::move(queue.front());
            queue.pop_front();
            ptr.AllocID(marker.AsHash());
        } else {
            ptr = Allocator::CreateAllocation(size, marker);
        }

        assert(allocMarkers_.count(marker.AsHash()) == 0
               || allocMarkers_.at(marker.AsHash()) == marker);
        allocMarkers_.emplace(marker.AsHash(), marker);
        stats_[marker.AsHash()].AllocationSize(size);

        return ptr;
    }

    void ReturnAlloc(AllocationT alloc)
    {
        std::lock_guard<std::mutex> lm(m_);

        if (!cache_) return;

        if (alloc.AllocID() != 0)
            stats_[alloc.AllocID()].DeallocationSize(alloc.size());
        allocs_[alloc.size()].push_front(std::move(alloc));
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

        if (!stats_.empty())
        {
            PBLOG_INFO << "----Memory high water mark report for " << Allocator::description << "-----";
            Report(stats_);
            PBLOG_INFO << "------------------------------------------";
            PBLOG_INFO << "Note: Different filter stages consume memory at different times.  "
                       << "The sum of high water marks for individual high watermarks won't "
                       << "be quite the same as the high watermark for the application as a whole";
        }

        // Delete all allocations (and statistics) we're currently holding on to.
        allocs_.clear();
        stats_.clear();
        allocMarkers_.clear();
    }


    std::map<size_t, std::deque<AllocationT>> allocs_;
    std::map<size_t, AllocStat> stats_;
    std::map<size_t, AllocationMarker> allocMarkers_;

    std::mutex m_;
    bool cache_ = false;
};

template <typename HostAllocator>
class MongoCachedAllocator final : public IMongoCachedAllocator
{
    using HostManager = AllocationManager<HostAllocator>;
    using GpuManager = AllocationManager<GpuAllocator>;
public:
    MongoCachedAllocator(const AllocationMarker& defaultMarker)
        : defaultMarker_(defaultMarker)
    {}

    PacBio::Memory::SmartAllocation GetAllocation(size_t size) override
    {
        return GetAllocation(size, defaultMarker_);
    }
    PacBio::Memory::SmartAllocation GetAllocation(size_t size, const AllocationMarker& marker) override
    {
        static thread_local size_t counter = 0;
        counter++;
        auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
        Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());

        auto tmp = prof.CreateScopedProfiler(ManagedAllocations::HostAllocate);
        return HostManager::GetManager().GetAlloc(size, marker);
    }

    SmartDeviceAllocation GetDeviceAllocation(size_t size, const AllocationMarker& marker) override
    {
        static thread_local size_t counter = 0;
        counter++;
        auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
        Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());

        auto tmp = prof.CreateScopedProfiler(ManagedAllocations::GpuAllocate);
        return GpuManager::GetManager().GetAlloc(size, marker);
    }

    bool SupportsAllFlags(uint32_t flags) const override
    {
        return (flags & HostAllocator::supportedIAllocatorFlags) == flags;
    }

private:
    AllocationMarker defaultMarker_;
};

std::pair<AllocatorMode, std::unique_ptr<IMongoCachedAllocator>>& AllocInstance()
{
    using Data = std::pair<AllocatorMode, std::unique_ptr<IMongoCachedAllocator>>;
    static Data data{AllocatorMode::MALLOC,
                     std::make_unique<MongoCachedAllocator<MallocAllocator>>(AllocationMarker("Not Recorded"))};

    return data;
}

} // anon namespace

// Helper function, to help consolidate switches over AllocatorMode
// to a single location.  `f` is expected to be a function-like
// thing that accepts an AllocationManager as an argument
template <typename F>
void VisitHostManager(AllocatorMode mode, F&& f)
{
    switch (mode)
    {
    case AllocatorMode::MALLOC:
        {
            f(AllocationManager<MallocAllocator>::GetManager());
            return;
        }
    case AllocatorMode::CUDA:
        {
            f(AllocationManager<PinnedAllocator>::GetManager());
            return;
        }
    case AllocatorMode::HUGE_CUDA:
        {
            f(AllocationManager<HugePinnedAllocator>::GetManager());
            return;
        }
    }
    throw PBException("Unexpected AllocationMode");
}

void IMongoCachedAllocator::ReturnHostAllocation(PacBio::Memory::SmartAllocation alloc)
{
    if (alloc.size() == 0) return;
    static thread_local size_t counter = 0;
    counter++;
    auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
    Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
    auto tmp = prof.CreateScopedProfiler(ManagedAllocations::HostDeallocate);

    auto allocMode = static_cast<AllocatorMode>(alloc.AllocatorType());
    VisitHostManager(allocMode,
                 [alloc = std::move(alloc)](auto& manager) mutable
                 {
                     manager.ReturnAlloc(std::move(alloc));
                 });
}

void IMongoCachedAllocator::ReturnDeviceAllocation(SmartDeviceAllocation alloc)
{
    if (alloc.size() == 0) return;
    static thread_local size_t counter = 0;
    counter++;
    auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
    Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());

    auto tmp = prof.CreateScopedProfiler(ManagedAllocations::GpuDeallocate);
    AllocationManager<GpuAllocator>::GetManager().ReturnAlloc(std::move(alloc));
}

void SetGlobalAllocationMode(CachingMode caching, AllocatorMode alloc)
{
    auto& data = AllocInstance();
    data.first = alloc;
    data.second = CreateAllocator(alloc, AllocationMarker("Not Recorded"));

    if (caching == CachingMode::ENABLED)
    {
        EnableHostCaching(alloc);
        EnableGpuCaching();
    } else
    {
        DisableHostCaching(alloc);
        DisableGpuCaching();
    }
}

std::unique_ptr<IMongoCachedAllocator> CreateAllocator(AllocatorMode alloc, const AllocationMarker& marker)
{
    switch (alloc)
    {
    case AllocatorMode::MALLOC:
        {
            return std::make_unique<MongoCachedAllocator<MallocAllocator>>(marker);
        }
    case AllocatorMode::CUDA:
        {
            return std::make_unique<MongoCachedAllocator<PinnedAllocator>>(marker);
        }
    case AllocatorMode::HUGE_CUDA:
        {
            return std::make_unique<MongoCachedAllocator<HugePinnedAllocator>>(marker);
        }
    }
    throw PBException("Unsupported AllocatorMode");
}

IMongoCachedAllocator& GetGlobalAllocator()
{
    return *AllocInstance().second;
}

void EnableHostCaching(AllocatorMode mode)
{
    VisitHostManager(mode,
                 [](auto& manager)
                 {
                     manager.EnableCaching();
                 });
}
void EnableGpuCaching()
{
    AllocationManager<GpuAllocator>::GetManager().EnableCaching();
}

void DisableHostCaching(AllocatorMode mode)
{
    VisitHostManager(mode,
                 [](auto& manager)
                 {
                     manager.DisableCaching();
                 });
}
void DisableGpuCaching()
{
    AllocationManager<GpuAllocator>::GetManager().DisableCaching();
}

void DisableAllCaching()
{
    // I don't think this report has much value for normal runs,
    // but may be useful if profiling or diagnosing performance
    // issues, in which case we're probably using RelWithDebInfo
    // and this will turn on.
#ifndef NDEBUG
    Profiler::FinalReport();
#endif
    DisableHostCaching(AllocatorMode::MALLOC);
    DisableHostCaching(AllocatorMode::CUDA);
    DisableHostCaching(AllocatorMode::HUGE_CUDA);
    DisableGpuCaching();
}


}}} // ::PacBio::Cuda::Memory
