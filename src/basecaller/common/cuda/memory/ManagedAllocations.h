// Copyright (c) 2019-2021 Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS
#define PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS

#include <cstring>
#include <string>

#include <mongo-config.h>

#include <pacbio/datasource/SharedMemoryAllocator.h>
#include <pacbio/memory/SmartAllocation.h>
#include <pacbio/memory/IAllocator.h>
#include <pacbio/utilities/Finally.h>
#include <common/cuda/memory/SmartDeviceAllocation.h>

namespace PacBio {
namespace Cuda {
namespace Memory {

// Class used to give some sort of identity to an
// allocation, for the purpose of allocation tracking.
// It can be any string, though below a SOURCE_MARKER
// macro is provided for convenience, which will
// automatically create a marker of the form `file:line`
//
// Using the same marker for multiple allocations is fine
// as long as they make sense to bundle together in reported
// statistics
class AllocationMarker
{
public:
    AllocationMarker(const std::string& mark)
        : mark_(mark)
    {}

    bool operator==(const AllocationMarker& o) const
    {
        return mark_ == o.mark_;
    }

    // Retrieve string representation
    const std::string& AsString() const { return mark_; }
    // Retrieve hash representation.  Hash collisions are
    // technically possible, though hopefully unlikely, so
    // consumers of this function need to be aware of that
    size_t AsHash() const { return std::hash<std::string>{}(mark_); }

private:
    std::string mark_;
};

// Creates an AllocationMarker tied to the source code file and line number.
// Only uses the basename() of the source file name, or the whole __FILE__ if / is not found.
#define SOURCE_MARKER()                                                                 \
        PacBio::Cuda::Memory::AllocationMarker(                                         \
                [](){const char*x=strrchr(__FILE__,'/'); return std::string(x?x:__FILE__);}() \
                + ":" + std::to_string(__LINE__))
// Note: POSIX/GNU versions of basename require the path argument to be read/write, so it can't be used.

/// Extension of the IAllocator interface, which adds the notion of a GPU
/// allocations (which for safety reasons are a fundamentally different
/// type from host allocations)
///
/// Children implementations can optionally use an allocation cache,
/// which is significantly faster than allocating/deallocating
/// individual allocations each time they are needed, as the CUDA
/// allocators seem to be painfully slow
///
/// \note Different instances of IMongoCachedAllocator may share the same allocation cache.
///       Certain things cannot be mixed (like huge pages and normal system allocations),
///       but otherwise the implementation can share allocation pools between instances,
///       as long as those instances are created with CacheMode::GLOBAL_CACHE.
/// \note The allocation caching scheme is currently very simple.  Allocations of irregular
///       size will probably result never be recycled, and they will live until the cache
///       is manually cleared (probably not until the end of execution).  In this situation
///       it will effectively appear to be a memory leak.
///
/// \warn Allocations must not live longer than the KestrelAllocator that produced them, and
///       if that happens then said allocations will be explicitly leaked.  This shouldn't
///       be a problem for vanilla usage, but it's not unknown for certain parts of the core
///       analyzer to set up data structures in static variables for common usage.  Those
///       static variables should be explicitly cleared out when analysis ends rather than
///       relying on static teardown after the main function exits.
class KestrelAllocator : public PacBio::Memory::IAllocator
{
public:
    using IAllocator::GetAllocation;

    virtual PacBio::Memory::SmartAllocation
    GetAllocation(size_t count, const AllocationMarker& marker) = 0;

    virtual SmartDeviceAllocation
    GetDeviceAllocation(size_t count, const AllocationMarker& marker) = 0;
};

/// Supported flavors of host allocations for mongo
enum class AllocatorMode
{
    /// Simple malloc/free implementation
    MALLOC,
    /// Allocated memory as "Pinned", which significantly speeds
    /// up Host <-> Gpu data transfers.
    /// ("Pinning" is effectively just page locking the memory,
    /// and then letting the cuda runtime know that said memory
    /// is page locked)
    CUDA,
    /// Sets up memory that is both shared huge pages for
    /// compatability with the WX2, as well as pinned for
    /// efficient upload to the GPU.
    SHARED_MEMORY_HUGE_CUDA
};

/// Controls the caching scheme to be used
enum class CacheMode
{
    /// Allocations are not cached
    DISABLED,
    /// All KestrelAllocators (with the same AllocatorMode) that
    /// use this mode will share the same cache
    GLOBAL_CACHE,
    /// Use a cache specific to this KestrelAllocator
    PRIVATE_CACHE
};

/// Sets the global allocation mode.  This is controlled
/// on an application level, to better faciliate scenarios
/// such as running a host-only version on a computer without
/// a GPU (where allocating pinned host memory would fail)
///
/// The initial configuration is CacheMode::DISABLED and
/// AllocatorMode::Malloc.  Due to static lifetime issues,
/// it is not safe to have caching enabled during static teardown,
/// as the Cuda Runtime may already be destroyed by the time we
/// free any device memory still in our caches, which would
/// cause a hard crash of the application.  Due to this, the
/// return value of this function is a `Finally` object that will
/// reset the global settings back to the original default.
///
/// Note: This function is not thread safe, and should not be
///       called concurrently either with itself or with
///       GetGlobalAllocator();
///
/// \return a Finally object that will reset the global allocation
///         settings back to default upon the end of scope.
[[nodiscard]] PacBio::Utilities::Finally
SetGlobalAllocationMode(CacheMode cacheMode, AllocatorMode alloc);

/// Gets globabl KestrelAllocator, using whatever mode
/// the application as a whole has been configured to use
///
/// Note: It is safe for multiple threads to call this function
///       concurrently, just none of those calls can be
///       concurrent with SetGlobalAllocationMode
KestrelAllocator& GetGlobalAllocator();

/// Logs a report about all memory usage statistics
void ReportAllMemoryStats();

/// Returns a string summarizing the shared memory, suitable for a frequent log entry
std::string SummarizeSharedMemory();

/// Creates a KestrelAllocator with a Malloc based implementation,
/// primarily for host-only code, or code that doesn't care about
/// performance.  It can still do GPU allocations, but it does not
/// use "Pinned" memory on the host, which is required for efficient
/// data transfers
std::unique_ptr<KestrelAllocator>
CreateMallocAllocator(const AllocationMarker& defaultMarker,
                      CacheMode cacheMode = CacheMode::DISABLED);

/// Creates a KestrelAllocator that uses Pinned host memory.
/// This is the preferred allocator for uploading data to the
/// GPU, as said allocations will transfer significantly more
/// efficiently.
std::unique_ptr<KestrelAllocator>
CreatePinnedAllocator(const AllocationMarker& defaultMarker,
                      CacheMode cacheMode = CacheMode::DISABLED);

/// Creates a KestrelAllocator that both uses memory that is
/// both Pinned, and also uses shared huge pages.  The latter
/// is required for interfacing with things like the WX2.
std::unique_ptr<KestrelAllocator>
CreateSharedHugePinnedAllocator(const AllocationMarker& defaultMarker,
                                std::unique_ptr<DataSource::SharedMemoryAllocator> sharedAlloc,
                                CacheMode cacheMode = CacheMode::DISABLED);
}}} // ::PacBio::Cuda::Memory

#endif // PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS
