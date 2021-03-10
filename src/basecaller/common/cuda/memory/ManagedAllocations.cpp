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

/////////////////////////////////////////
// Some backend allocators that an be
// plugged into the caching framework
/////////////////////////////////////////

struct MallocAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags = IAllocator::ALIGN_64B;
    static constexpr const char* description = "Standard Host Memory";

    static void* allocate(size_t size)
    {
        return aligned_alloc(64, size);
    }
    static void deallocate(void* ptr)
    {
        free(ptr);
    }
};

struct PinnedAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags =
          IAllocator::CUDA_MEMORY
        | IAllocator::ALIGN_64B;
    static constexpr const char* description = "Pinned Host Memory";

    static void* allocate(size_t size)
    {
        return CudaRawMallocHost(size);
    }
    static void deallocate(void* ptr)
    {
        CudaFreeHost(ptr);
    }
};

struct HugePinnedAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags =
          IAllocator::CUDA_MEMORY
        | IAllocator::HUGEPAGES
        | IAllocator::ALIGN_64B;

    static constexpr const char* description = "Pinned HugePage Host Memory";

    static void* allocate(size_t size)
    {
        return Allocator().Allocate(size);
    }

    static void deallocate(void* ptr)
    {
        Allocator().Deallocate(ptr);
    }

private:
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

    static void* allocate(size_t size)
    {
        return CudaRawMalloc(size);
    }
    static void deallocate(void* ptr)
    {
        CudaFree(ptr);
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
    AllocationManager()
    {
        // We're doing something somewhat subtle here.  AllocationManager is
        // used with a static storage duration, but it's members include memory
        // allocations that in turn rely on other data structures with static
        // storage as well.  The most notable of these is the cuda runtime
        // itself.  If we're not careful about the lifetime of static objects,
        // we will potentially crash during application teardown, when we try
        // to give memory back to the cuda runtime that is no longer alive.
        //
        // The destruction order of static/global objects is the reverse of
        // the *completion* of their constructors.  By creating and destroying
        // an allocation here during the ctor, we ensure that all static/global
        // infrastructure required to free allocations will live longer than
        // this object, making destruction of the contained allocations safe.
        Allocator::deallocate(Allocator::allocate(1));
    }

    AllocationManager(const AllocationManager&) = delete;
    AllocationManager(AllocationManager&&) = delete;
    AllocationManager& operator=(const AllocationManager&) = delete;
    AllocationManager& operator=(AllocationManager&&) = delete;

    struct Deleter
    {
        void operator()(void* ptr)
        {
            Allocator::deallocate(ptr);
        }
    };
    using Ptr_t = std::unique_ptr<void, Deleter>;
public:

    // Singleton access function.  Static variables in a function have
    // a well defined and sane initialization guarantee, while actual
    // global variables do not (since the ordering of initialization
    // between different translation units is indeterminate)
    // We're returning a shared_pointer here so that judicious useage
    // of weak_ptr can help detect lifetime issues resulting from static
    // teardown order.
    static std::shared_ptr<AllocationManager> GetManager()
    {
        static auto manager = std::shared_ptr<AllocationManager>(new AllocationManager{});
        return manager;
    };

    ~AllocationManager()
    {
        if (cache_) FlushPools();
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

    void* GetAlloc(size_t size, const AllocationMarker& marker)
    {
        if (size == 0) return nullptr;

        std::lock_guard<std::mutex> lm(m_);
        assert(allocMarkers_.count(marker.AsHash()) == 0
               || allocMarkers_.at(marker.AsHash()) == marker);
        allocMarkers_.emplace(marker.AsHash(), marker);
        stats_[marker.AsHash()].AllocationSize(size);

        auto& queue = allocs_[size];
        if (queue.size() > 0)
        {
            auto alloc = std::move(queue.front());
            queue.pop_front();
            return alloc.release();
        } else {
            return Allocator::allocate(size);
        }
    }

    void ReturnAlloc(void* alloc, size_t size, size_t allocID)
    {
        Utilities::Finally f([&]{ if(alloc) Allocator::deallocate(alloc);});
        std::lock_guard<std::mutex> lm(m_);

        if (cache_)
        {
            if (allocID != 0)
                stats_[allocID].DeallocationSize(size);
            allocs_[size].push_front(Ptr_t{alloc});
            alloc = nullptr;
        }
    }

    void FlushPools()
    {
        std::lock_guard<std::mutex> lm(m_);

        FlushPoolsImpl();
    }

    void Report()
    {
        std::lock_guard<std::mutex> lm(m_);

        ReportImpl();
    }

private:
    void ReportImpl()
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
    }
    // Clears out all allocation pools (and generates a usage report)
    void FlushPoolsImpl()
    {
        ReportImpl();

        // Delete all allocations (and statistics) we're currently holding on to.
        allocs_.clear();
        stats_.clear();
        allocMarkers_.clear();
    }


    std::map<size_t, std::deque<Ptr_t>> allocs_;
    std::map<size_t, AllocStat> stats_;
    std::map<size_t, AllocationMarker> allocMarkers_;

    std::mutex m_;
    bool cache_ = false;
};

template <typename HostAllocator>
class MongoCachedAllocator final : public IMongoCachedAllocator, private detail::DataManager
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
        (void)tmp;
        auto mngr = HostManager::GetManager();
        return PacBio::Memory::SmartAllocation{size,
            // Storing mngr as a weak pointer in these lambdas to help
            // sniff out static teardown issues.  It's a bit of cautious
            // paranoia for the first lambda since the current implemntation
            // evaluates it immediately, but it's more important for the second
            // when the allocation might be returned after the Manager is already
            // dead.  There is no robust promise of recovery, but we'll at least
            // avoid some obvious and definite sources of UB.
            [mngr = std::weak_ptr<AllocationManager<HostAllocator>>{mngr},
             &marker](size_t sz){
                auto manager = mngr.lock();
                if (manager)
                    return manager->GetAlloc(sz, marker);
                else
                    return static_cast<void*>(nullptr);
            },
            [mngr = std::weak_ptr<AllocationManager<HostAllocator>>{mngr},
             size,
             id = marker.AsHash()]
            (void* ptr){
                static thread_local size_t counter = 0;
                counter++;
                auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
                Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
                auto tmp = prof.CreateScopedProfiler(ManagedAllocations::HostDeallocate);
                (void)tmp;

                auto m = mngr.lock();
                if (m)
                {
                    m->ReturnAlloc(ptr, size, id);
                } else {
                    // Logging might be risky to do??  Tossing in an assert to at least
                    // give a convenient breakpoint for a debugger before we do anything
                    // blatantly UB.
                    assert(false);
                    PBLOG_ERROR << "No allocation manager, things are torn down?"
                                << " Intentionally leaking allocation...";
                }
            }};
    }

    SmartDeviceAllocation GetDeviceAllocation(size_t size, const AllocationMarker& marker) override
    {
        static thread_local size_t counter = 0;
        counter++;
        auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
        Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());

        auto tmp = prof.CreateScopedProfiler(ManagedAllocations::GpuAllocate);
        (void)tmp;
        auto mngr = GpuManager::GetManager();
        return SmartDeviceAllocation{size,
            // Storing mngr as a weak pointer in these lambdas to help
            // sniff out static teardown issues.  It's a bit of cautious
            // paranoia for the first lambda since the current implemntation
            // evaluates it immediately, but it's more important for the second
            // when the allocation might be returned after the Manager is already
            // dead.  There is no robust promise of recovery, but we'll at least
            // avoid some obvious and definite sources of UB.
            [mngr = std::weak_ptr<AllocationManager<GpuAllocator>>{mngr},
             &marker](size_t sz){
                auto manager = mngr.lock();
                if (manager)
                    return manager->GetAlloc(sz, marker);
                else
                    return static_cast<void*>(nullptr);
            },
            [mngr = std::weak_ptr<AllocationManager<GpuAllocator>>{mngr},
             size,
             id = marker.AsHash()](void* ptr){
                static thread_local size_t counter = 0;
                counter++;
                auto mode = (counter < 1000) ? Profiler::Mode::IGNORE : Profiler::Mode::OBSERVE;
                Profiler prof(mode, 6.0f, std::numeric_limits<float>::max());
                auto tmp = prof.CreateScopedProfiler(ManagedAllocations::GpuDeallocate);
                (void)tmp;

                auto m = mngr.lock();
                if (m)
                {
                    m->ReturnAlloc(ptr, size, id);
                } else {
                    // Logging might be risky to do??  Tossing in an assert to at least
                    // give a convenient breakpoint for a debugger before we do anything
                    // blatantly UB.
                    assert(false);
                    PBLOG_ERROR << "No allocation manager, things are torn down?"
                                << " Intentionally leaking allocation...";
                }
            },
            DataKey()};
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
    // Putting this in a small lambda to avoid repeating the
    // null check.  I can't imagine this function will get called
    // after the static managers are destroyed, but might as well
    // be safe;
    auto Invoke = [&](auto&& mgr)
    {
        assert(mgr);
        f(*mgr);
    };
    switch (mode)
    {
    case AllocatorMode::MALLOC:
        {
            Invoke(AllocationManager<MallocAllocator>::GetManager());
            return;
        }
    case AllocatorMode::CUDA:
        {
            Invoke(AllocationManager<PinnedAllocator>::GetManager());
            return;
        }
    case AllocatorMode::HUGE_CUDA:
        {
            Invoke(AllocationManager<HugePinnedAllocator>::GetManager());
            return;
        }
    }
    throw PBException("Unexpected AllocationMode");
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

void ReportAllMemoryStats()
{
    auto Report = [](auto&& mgr)
    {
        if (mgr) mgr->Report();
    };

    Report(AllocationManager<MallocAllocator>::GetManager());
    Report(AllocationManager<PinnedAllocator>::GetManager());
    Report(AllocationManager<HugePinnedAllocator>::GetManager());
    Report(AllocationManager<GpuAllocator>::GetManager());
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
    auto mngr = AllocationManager<GpuAllocator>::GetManager();
    mngr->EnableCaching();
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
    auto mngr = AllocationManager<GpuAllocator>::GetManager();
    mngr->DisableCaching();
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
