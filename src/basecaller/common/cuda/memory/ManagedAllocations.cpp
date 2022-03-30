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
#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/HugePage.h>
#include <pacbio/datasource/SharedMemoryAllocator.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>
#include <pacbio/text/String.h>

SMART_ENUM(ManagedAllocations, HostAllocate, GpuAllocate, HostDeallocate, GpuDeallocate);
using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<ManagedAllocations>;

namespace PacBio {
namespace Cuda {
namespace Memory {

namespace {

using namespace PacBio::Memory;
using namespace PacBio::DataSource;

// Helper class to keep track of memory high water marks
class MemUsageStat
{
public:
    void AllocationSize(size_t size)
    {
        currentBytes_ += size;
        past_.peakBytes = std::max(currentBytes_, past_.peakBytes);
        past_.recentPeakBytes = std::max(currentBytes_, past_.recentPeakBytes);
    }
    void DeallocationSize(size_t size)
    {
        assert(size <= currentBytes_);
        currentBytes_ -= size;
        past_.recentMinBytes = std::min(currentBytes_, past_.recentMinBytes);
    }

    struct PastUsage
    {
        size_t peakBytes{0};
        size_t recentPeakBytes{0};
        size_t recentMinBytes{0};
    };

    // Returns the memory usage statistics between now and the previous call.
    // That does mean that this call effectively resets all statistics marked
    // as "recent"
    PastUsage PastMemUsage()
    {
        return std::exchange(past_, {past_.peakBytes, currentBytes_, currentBytes_});
    }

    size_t CurrentBytes() const { return currentBytes_; }
private:
    size_t currentBytes_{0};
    PastUsage past_;
};

/////////////////////////////////////////
// Some backend allocators that an be
// plugged into the caching framework
/////////////////////////////////////////

struct MallocAllocator
{
    static constexpr uint32_t supportedIAllocatorFlags = IAllocator::ALIGN_64B;
    static constexpr const char* description = "Standard Host Memory";

    static std::string StaticName() { return "MallocAllocator";}

    void* allocate(size_t size)
    {
        static constexpr uint32_t alignment = 64;
        return aligned_alloc(64, (size + alignment - 1) / alignment * alignment);
    }
    void deallocate(void* ptr)
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

    static std::string StaticName() { return "HugePinnedPinnedAllocatorAllocator";}

    void* allocate(size_t size)
    {
        try
        {
            return CudaRawMallocHost(size);
        }
        // Specifically *only* catching memory allocation failures.  Other exceptions
        // that we're not equiped to handle may be thrown, due the asynchronous nature
        // of cuda error reporting, and those we'll just let bubble upwards.
        catch (const CudaMemException& ex)
        {
            PBLOG_ERROR << "CudaRawMallocHost(size= " << size << ");";
            throw;
        }
    }
    void deallocate(void* ptr)
    {
        CudaFreeHost(ptr);
    }
};

class SharedHugePinnedAllocator
{
public:
    SharedHugePinnedAllocator(std::unique_ptr<SharedMemoryAllocator> sharedAllocator)
        : slicer_{1<<30, 1, Helper{std::move(sharedAllocator)}}
    {}

    static constexpr uint32_t supportedIAllocatorFlags =
          IAllocator::CUDA_MEMORY
        | IAllocator::HUGEPAGES
        | IAllocator::ALIGN_64B
        | IAllocator::ALIGN_256B
        | IAllocator::SHARED_MEMORY;

    static constexpr const char* description = "Shared Pinned HugePage Host Memory";

    static std::string StaticName() { return "SharedHugePinnedAllocator";}

    void* allocate(size_t size)
    {
        try
        {
            return slicer_.Allocate(size);
        }
        // Specifically *only* catching memory allocation failures.  Other exceptions
        // that we're not equiped to handle may be thrown, due the asynchronous nature
        // of cuda error reporting, and those we'll just let bubble upwards.
        catch (const CudaMemException& ex)
        {
            PBLOG_ERROR << "SharedHugePinnedAllocator failed to satisify an allocation request. ";
            PBLOG_ERROR << "Requested allocation size in bytes was: " << size;
            PBLOG_ERROR << "Dumping current usage information for the AllocationSlicer...";
            const auto& storage = slicer_.PeekStorage();
            for (const auto& kv : storage)
            {
                PBLOG_ERROR << "Base Address: " << kv.first;
                PBLOG_ERROR << "    Capacity/Size: " << kv.second.capacity() << " / " << kv.second.size();
                PBLOG_ERROR << "    Active Allocation Count" << kv.second.ActiveCount();
            }
            throw;
        }
    }

    void deallocate(void* ptr)
    {
        slicer_.Deallocate(ptr);
    }

private:
    // Helper class, to do the extra work required to unify huge
    // page allocations with the cuda runtime
    struct Helper
    {
        Helper(std::unique_ptr<SharedMemoryAllocator> allocator)
            : sharedAlloc_(std::move(allocator))
        {}

        void* Malloc(size_t size)
        {
            if(size % (1ull<<30) != 0)
                throw PBException("Implementation bug!  Expected request to "
                                  "be an integer number of 1GB Huge Pages");
            auto ptr = sharedAlloc_->Allocate(size);
            if((reinterpret_cast<uint64_t>(ptr) & ((1ull << 30) -1)) != 0)
                throw PBException("Implementation bug!  SharedMemoryAllocator "
                                  "was expected to return a 1GB aligned allocation");

            try
            {
                // This function both page-locks the allocation,
                // and registers the address range with the cuda
                // runtime
                CudaHostRegister(ptr, size);
                return ptr;
            }
            catch (const CudaMemException& ex)
            {
                PBLOG_ERROR << "Failed to register address with cuda runtime.  Addr - count: " << ptr << " - " << size;
                PBLOG_ERROR << "SharedMemoryAllocator usage information" << sharedAlloc_->ReportStatus();
                throw;
            }
        }

        void Free(void* ptr)
        {
            // This function unlocks the page, and updates
            // the cuda runtime bookkeeping to reflect that
            CudaHostUnregister(ptr);
            sharedAlloc_->Deallocate(ptr);
        }

        std::unique_ptr<SharedMemoryAllocator> sharedAlloc_;
    };

private:
    AllocationSlicer<Helper, 256> slicer_;
};

struct GpuAllocator
{
    static constexpr const char* description = "Gpu Memory";

    void* allocate(size_t size)
    {
        try
        {
            return CudaRawMalloc(size);
        }
        // Specifically *only* catching memory allocation failures.  Other exceptions
        // that we're not equiped to handle may be thrown, due the asynchronous nature
        // of cuda error reporting, and those we'll just let bubble upwards.
        catch (const CudaMemException& ex)
        {
            PBLOG_ERROR << "CudaRawMalloc(size= " << size << ");";
            throw;
        }
    }
    void deallocate(void* ptr)
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
    // Use ProcureCache instead.
    template <typename...Args, std::enable_if_t<std::is_constructible_v<Allocator, Args...>, int> = 0>
    AllocationManager(CacheMode cachingMode, Args&&... args)
        : allocator_(std::forward<Args>(args)...)
        , cache_(cachingMode == CacheMode::DISABLED ? false : true)
    {}

public:

    /// static generator function for obtaining an AllocationManager
    /// This is used instead of a regular constructor for two reasons:
    /// 1. Allow the retrieval of a pre-existing cache when CacheMode::GLOBAL_CACHE is used
    /// 2. Keep track of all caches alive, so that their usage and memory statistics can
    ///    be observed
    template <typename... Args, std::enable_if_t<std::is_constructible_v<Allocator, Args...>, int> = 0>
    static std::shared_ptr<AllocationManager> ProcureCache(CacheMode mode, Args&&... args)
    {
        std::lock_guard<std::mutex> lm(staticMutex_);

        std::shared_ptr<AllocationManager> ret;

        if (mode == CacheMode::GLOBAL_CACHE)
        {
            ret = globalCache_.lock();
        }

        if (!ret)
        {
            ret.reset(new AllocationManager(mode, std::forward<Args>(args)...));
            refList_.push_back(ret);
            if (mode == CacheMode::GLOBAL_CACHE)
                globalCache_ = ret;
        }

        return ret;
    }

    /// Note: Has a minor side effect where the static vector of weak_ptrs
    ///       will have it's empty members pruned
    ///
    /// \return a vector of shared_ptr to all current AllocationManager instances
    static std::vector<std::shared_ptr<AllocationManager>> GetAllRefs()
    {
        std::lock_guard<std::mutex> lm(staticMutex_);

        std::vector<std::shared_ptr<AllocationManager>> ret;
        ret.reserve(refList_.size());
        for (auto& weak : refList_)
        {
            auto shared = weak.lock();
            if (shared) ret.push_back(shared);
        }

        //prune empty weak_ptr instances while we are here
        refList_.erase(std::remove_if(refList_.begin(),
                                      refList_.end(),
                                      [](auto& weak) { return weak.expired(); }),
                       refList_.end());


        return ret;
    }

    AllocationManager(const AllocationManager&) = delete;
    AllocationManager(AllocationManager&&) = delete;
    AllocationManager& operator=(const AllocationManager&) = delete;
    AllocationManager& operator=(AllocationManager&&) = delete;

private:
    struct Deleter
    {
        void operator()(void* ptr)
        {
            assert(allocator_);
            allocator_->deallocate(ptr);
        }
        Allocator* allocator_{nullptr};
    };
    using Ptr_t = std::unique_ptr<void, Deleter>;
public:

    ~AllocationManager()
    {
        if (cache_) FlushPools();
    }

    void* GetAlloc(size_t size, const AllocationMarker& marker)
    {
        if (size == 0) return nullptr;

        std::lock_guard<std::mutex> lm(instanceMutex_);
        assert(allocMarkers_.count(marker.AsHash()) == 0
               || allocMarkers_.at(marker.AsHash()) == marker);
        allocMarkers_.emplace(marker.AsHash(), marker);
        stats_[marker.AsHash()].AllocationSize(size);
        fullStats_.AllocationSize(size);

        auto& queue = allocs_[size];
        if (queue.size() > 0)
        {
            auto alloc = std::move(queue.front());
            queue.pop_front();
            cacheStats_.DeallocationSize(size);
            return alloc.release();
        } else {
            try
            {
                return allocator_.allocate(size);
            } catch (const CudaMemException& ex)
            {
                PBLOG_ERROR << "Cuda runtime had an error while allocating memory.";
                PBLOG_ERROR << "Current total mem usage, including cache, is " << cacheStats_.CurrentBytes() + fullStats_.CurrentBytes();
                PBLOG_ERROR << "Current usage report for this AllocationManager:";
                Logging::LogStream ls(Logging::LogLevel::ERROR);;
                ls << ReportImpl();
                throw;
            }
        }

    }

    void ReturnAlloc(void* ptr, size_t size, size_t allocID)
    {
        Utilities::Finally f([&](){ if(ptr) allocator_.deallocate(ptr);});
        std::lock_guard<std::mutex> lm(instanceMutex_);

        stats_[allocID].DeallocationSize(size);
        fullStats_.DeallocationSize(size);

        if (cache_)
        {
            cacheStats_.AllocationSize(size);
            allocs_[size].push_front(Ptr_t{ptr, {&allocator_}});
            ptr = nullptr;
        }
    }

    void FlushPools()
    {
        std::lock_guard<std::mutex> lm(instanceMutex_);

        FlushPoolsImpl();
    }

    /// Generates report, to show what sections of code are allocating
    /// the most memory
    std::string Report()
    {
        std::lock_guard<std::mutex> lm(instanceMutex_);

        return ReportImpl();
    }

    /// Returns the sum of current bytes
    size_t CurrentBytes() const
    {
        return fullStats_.CurrentBytes() + cacheStats_.CurrentBytes();
    }

private:
    std::string ReportImpl()
    {
        if (stats_.empty()) return "";

        struct TaggedStats
        {
            std::string name;
            MemUsageStat::PastUsage stats;
        };
        // All the stats, sorted by decreasing recent
        // memory usage
        std::vector<TaggedStats> sortedStats;
        for (auto& kv : stats_)
        {
            sortedStats.push_back({
                    allocMarkers_.at(kv.first).AsString(),
                    kv.second.PastMemUsage()
            });
        }

        std::sort(sortedStats.begin(),
                  sortedStats.end(),
                  [](const auto& l, const auto& r)
                  {return l.stats.recentPeakBytes > r.stats.recentPeakBytes; });

        const auto peakSum = std::accumulate(sortedStats.begin(),
                                             sortedStats.end(),
                                             0ull,
                                             [](const auto& a, const auto& b)
                                             { return a+b.stats.peakBytes; });

        const auto recentPeakSum = std::accumulate(sortedStats.begin(),
                                                   sortedStats.end(),
                                                   0ull,
                                                   [](const auto& a, const auto& b)
                                                   { return a+b.stats.recentPeakBytes; });

        // This is a measure of any allocations that now seem to reside permanently
        // in the cache, and aren't really being used anymore.  This is a side effect
        // of a memory usage spike, and due to the simplicitly of the current caching
        // scheme, this is memory that won't be returned to the system until the
        // cache is destroyed
        const auto recentFallow = cacheStats_.PastMemUsage().recentMinBytes;
        const auto totalAllocs = fullStats_.PastMemUsage().peakBytes;

        auto PrettyMemNumber = [](auto& stream, float f) -> decltype(auto)
        {
            constexpr auto KiB = static_cast<float>(1<<10);
            constexpr auto MiB = static_cast<float>(1<<20);
            constexpr auto GiB = static_cast<float>(1<<30);

            stream << std::setw(4);

            if (f >= GiB) stream << f / GiB << " GiB";
            else if (f >= MiB) stream << f / MiB << " MiB";
            else if (f >= KiB) stream << f / KiB << " KiB";
            else stream << f << "B";

            return stream;
        };

        std::stringstream msg;
        msg << std::setprecision(3);

        msg << "----Memory usage report for " << Allocator::description << "-----\n"
            << "Total mem usage: ";
        PrettyMemNumber(msg, totalAllocs) << " \n"
            << "All Time High Water Marks sum: ";
        PrettyMemNumber(msg, peakSum) << " (Savings: "
            << 100.0f - 100.0f * totalAllocs / peakSum << "%)\n"
            << "Recent High Water Marks Sum: ";
        PrettyMemNumber(msg, recentPeakSum) << " (Savings: "
            << 100.0f - 100.0f * totalAllocs / recentPeakSum << "%)\n"
            << "Unused Cache: ";
        PrettyMemNumber(msg, recentFallow) << " (Waste: "
            << 100.0f * recentFallow / totalAllocs << "%)\n"
            << "High Water Breakdown:\n";

        for (const auto& val : sortedStats)
        {
            msg << "\t" << std::setw(5) << 100.0f * val.stats.recentPeakBytes / recentPeakSum << "% : ";
            PrettyMemNumber(msg, val.stats.recentPeakBytes) << " ("
                << std::setw(5) << 100.0f * val.stats.recentPeakBytes / val.stats.peakBytes << "% Of Peak) : "
                << val.name << "\n";
        }

        msg << "------------------------------------------\n";
        msg << "Note: Different filter stages consume memory at different times.  "
            << "The sum of high water marks for individual high watermarks is not expected to "
            << "be quite the same as the high watermark for the application as a whole\n";

        return msg.str();
    }

    // Clears out all allocation pools (and generates a usage report)
    void FlushPoolsImpl()
    {
        Logging::LogStream ls(Logging::LogLevel::INFO);;
        ls << ReportImpl();

        // Delete all allocations (and statistics) we're currently holding on to.
        allocs_.clear();
        stats_.clear();
        allocMarkers_.clear();
    }

    // We do have an explicit clearing of the cache before member variables are
    // destroyed, but I'm still listing this first since entries into allocs_
    // below maintain a reference to the allocator_ here
    Allocator allocator_;

    std::map<size_t, std::deque<Ptr_t>> allocs_;
    std::map<size_t, MemUsageStat> stats_;
    std::map<size_t, AllocationMarker> allocMarkers_;

    MemUsageStat cacheStats_;
    MemUsageStat fullStats_;

    std::mutex instanceMutex_;
    bool cache_;

    // Some static data to keep track of all the managers currently in existence.
    // We're using weak_ptr here specifically because we want to observe, but
    // not prolong the lifetime of things
    static std::mutex staticMutex_;
    static std::weak_ptr<AllocationManager> globalCache_;
    static std::vector<std::weak_ptr<AllocationManager>> refList_;
};

template <typename T>
std::mutex AllocationManager<T>::staticMutex_;
template <typename T>
std::weak_ptr<AllocationManager<T>> AllocationManager<T>::globalCache_;
template <typename T>
std::vector<std::weak_ptr<AllocationManager<T>>> AllocationManager<T>::refList_;

template <typename HostAllocator>
class CachedAllocatorImpl final : public KestrelAllocator, private detail::DataManager
{
    using HostManager = AllocationManager<HostAllocator>;
    using GpuManager = AllocationManager<GpuAllocator>;
public:
    template <typename...Args, std::enable_if_t<std::is_constructible_v<HostManager, CacheMode, Args...>, int> = 0>
    CachedAllocatorImpl(const AllocationMarker& defaultMarker, CacheMode cacheMode, Args&&... args)
        : defaultMarker_(defaultMarker)
        , hostManager_(HostManager::ProcureCache(cacheMode, std::forward<Args>(args)...))
        , gpuManager_(GpuManager::ProcureCache(cacheMode))
    {}

    std::string Name() const override
    {
        return "MongoCachedAllocator<" + HostAllocator::StaticName() + ">";
    }

    std::string ReportStatus() const override
    {
        return hostManager_->Report() + "\n" + gpuManager_->Report();
    }

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
        return PacBio::Memory::SmartAllocation{size,
           // Storing hostManager_ as a weak pointer in these lambdas to help
           // sniff out lifetime issues.  It's a bit of cautious paranoia
           // for the first lambda, since the current implementation
           // evaluates it immediately, but it's more important for the second.
           // If any allocations are returned after the manager is already
           // destroyed then we will (try to) log the issue and explicitly
           // leak the memory.  Since a likely cause for this scenario is that
           // someone used the allocation inside a static object, we may well
           // be in the static teardown phase after main exits, and there isn't
           // much more we can reliably do, especially since if we try to free
           // any cuda allocations after the cuda runtime is destroyed, we'll
           // probably get a hard crash
           [mngr = std::weak_ptr<AllocationManager<HostAllocator>>{hostManager_},
            &marker](size_t sz){
               auto manager = mngr.lock();
               if (manager)
                   return manager->GetAlloc(sz, marker);
               else
                   return static_cast<void*>(nullptr);
           },
           [mngr = std::weak_ptr<AllocationManager<HostAllocator>>{hostManager_},
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
        return SmartDeviceAllocation{size,
            // Storing gpuManager_ as a weak pointer in these lambdas to help
            // sniff out lifetime issues.  It's a bit of cautious paranoia
            // for the first lambda, since the current implementation
            // evaluates it immediately, but it's more important for the second.
            // If any allocations are returned after the manager is already
            // destroyed then we will (try to) log the issue and explicitly
            // leak the memory.  Since a likely cause for this scenario is that
            // someone used the allocation inside a static object, we may well
            // be in the static teardown phase after main exits, and there isn't
            // much more we can reliably do, especially since if we try to free
            // any cuda allocations after the cuda runtime is destroyed, we'll
            // probably get a hard crash
            [mngr = std::weak_ptr<AllocationManager<GpuAllocator>>{gpuManager_},
             &marker](size_t sz){
                auto manager = mngr.lock();
                if (manager)
                    return manager->GetAlloc(sz, marker);
                else
                    return static_cast<void*>(nullptr);
            },
            [mngr = std::weak_ptr<AllocationManager<GpuAllocator>>{gpuManager_},
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
    std::shared_ptr<HostManager> hostManager_;
    std::shared_ptr<GpuManager> gpuManager_;
};

struct GlobalModeInfo
{
    // We'll set this to true whenever we change to a new setting, so that we can
    // detect and error out if there are nested calls to SetGlobalAllocationMode.
    // For simplicity we're just not going to support that.
    bool customSettings = false;
    AllocatorMode mode = AllocatorMode::MALLOC;
    std::unique_ptr<KestrelAllocator> allocator =
        std::make_unique<CachedAllocatorImpl<MallocAllocator>>(std::string{"Not Recorded"},
                                                               CacheMode::DISABLED);
};

GlobalModeInfo& GetGlobalModeInfo()
{
    static GlobalModeInfo data;
    return data;
}

} // anon namespace

PacBio::Utilities::Finally SetGlobalAllocationMode(CacheMode cacheMode, AllocatorMode memType)
{
    auto& data = GetGlobalModeInfo();

    if (data.customSettings)
        throw PBException("Nested calls to SetGlobalAllocatoinMode are not supported. "
                          "We can only change the settings when they are currently at "
                          "the default");

    data.customSettings = true;

    AllocationMarker loc{"Not Recorded"};
    switch(memType)
    {
    case AllocatorMode::MALLOC:
        {
            data.allocator = std::make_unique<CachedAllocatorImpl<MallocAllocator>>(loc, cacheMode);
            break;
        }
    case AllocatorMode::CUDA:
        {
            data.allocator = std::make_unique<CachedAllocatorImpl<PinnedAllocator>>(loc, cacheMode);
            break;
        }
    case AllocatorMode::SHARED_MEMORY_HUGE_CUDA:
        {
            throw PBException("Cannot use SHARED_MEMORY_HUGE_CUDA as a global allocation mode");
            break;
        }
    default:
        throw PBException("Invalid AllocatorMode in SetGlobalAllocationMode");
    }
    data.mode = memType;

    return PacBio::Utilities::Finally([](){
        GetGlobalModeInfo() = GlobalModeInfo{};
    });
}

std::unique_ptr<KestrelAllocator>
CreateMallocAllocator(const AllocationMarker& defaultMarker,
                      CacheMode cachingMode)
{
    return std::make_unique<CachedAllocatorImpl<MallocAllocator>>(defaultMarker, cachingMode);
}

std::unique_ptr<KestrelAllocator>
CreatePinnedAllocator(const AllocationMarker& defaultMarker,
                      CacheMode cachingMode)
{
    return std::make_unique<CachedAllocatorImpl<PinnedAllocator>>(defaultMarker, cachingMode);
}

std::unique_ptr<KestrelAllocator>
CreateSharedHugePinnedAllocator(const AllocationMarker& defaultMarker,
                                std::unique_ptr<DataSource::SharedMemoryAllocator> sharedAlloc,
                                CacheMode cachingMode)
{
    return std::make_unique<CachedAllocatorImpl<SharedHugePinnedAllocator>>(defaultMarker, cachingMode, std::move(sharedAlloc));
}

KestrelAllocator& GetGlobalAllocator()
{
    return *GetGlobalModeInfo().allocator;
}

void ReportAllMemoryStats()
{

    Logging::LogStream ls(Logging::LogLevel::INFO);
    {
        auto caches = AllocationManager<MallocAllocator>::GetAllRefs();
        for (auto& cache : caches)
        {
            assert(cache);
            ls << cache->Report();
        }
    }
    {
        auto caches = AllocationManager<PinnedAllocator>::GetAllRefs();
        for (auto& cache : caches)
        {
            assert(cache);
            ls << cache->Report();
        }
    }
    {
        auto caches = AllocationManager<SharedHugePinnedAllocator>::GetAllRefs();
        for (auto& cache : caches)
        {
            assert(cache);
            ls << cache->Report();
        }
    }
    {
        auto caches = AllocationManager<GpuAllocator>::GetAllRefs();
        for (auto& cache : caches)
        {
            assert(cache);
            ls << cache->Report();
        }
    }
}

std::string SummarizeSharedMemory()
{
    std::ostringstream oss;
    oss << "SharedMemoryRecentPeakSum:";
    auto caches = AllocationManager<SharedHugePinnedAllocator>::GetAllRefs();
    bool first = true;
    for (auto& cache : caches)
    {
        assert(cache);
        if (!first) oss << "/";
        first = false;
        oss << PacBio::Text::String::FormatWithSI_Units(cache->CurrentBytes()) <<"B";
    }
    return oss.str();
}



}}} // ::PacBio::Cuda::Memory
