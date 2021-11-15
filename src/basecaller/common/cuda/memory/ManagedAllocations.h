// Copyright (c) 2019,2020 Pacific Biosciences of California, Inc.
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

#include <pacbio/PBException.h>
#include <pacbio/memory/SmartAllocation.h>
#include <pacbio/memory/IAllocator.h>
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
// Note: POSIX/GNU versions of baseline require the path argument to be read/write, so it can't be used.

/// Extension of the IAllocator interface, which adds the notion of a GPU
/// allocations (which for safety reasons are a fundamentally different
/// type from host allocations)
///
/// Children implementations will also use allocation caching,
/// which is significantly faster than allocating/deallocating
/// individual allocations each time they are needed as the CUDA
/// allocators seem to be painfully slow
///
/// \note Allocation caching can be disabled on an application wide basis via
///       SetGlobalAllocationMode
/// \note Different instances of IMongoCachedAllocator may share the same allocation cache.
///       Certain things cannot be mixed (like huge pages and normal system allocations),
///       but otherwise the implementation will share allocation pools between instances
/// \note The allocation caching scheme is currently very simple.  Allocations of irregular
///       size will probably result never be recycled, and they will live until the cache
///       is manually cleared (probably not until the end of execution).  In this situation
///       it will effectively appear to be a memory leak.
///
/// \warn There is an inherant risk in storing any allocation in a static variable and allowing
///       it to be destroyed automatically as part of runtime teardown.  It is notoriously
///       difficult to control the relative order of construction/destruction, and if you do
///       something like deallocate a cuda allocation after the cuda runtime itself is down
///       down, things will die gracelessly.  Great care has been done to make sure that the
///       caching infrastructure can always destroy itself without issue, but there is only so
///       much that can be done for any allocations that are created and outlive the allocation
///       cache.  We will be able to detect if we are destroying an allocation after the cache
///       is destroyed, but since the cache has static storage duration, if it's destroyed we
///       are already in the midst of application teardown.  At that point we don't even know
///       if calls to functions like cudaFree are safe to call.  Moral of the story is that if
///       you need to store data in static variables for visibility purposes or something,
///       you'll either need to either be very careful about the nuances of object lifetimes,
///       or have an explicit cleanup operation that destroys things before main exits.
class IMongoCachedAllocator : public PacBio::Memory::IAllocator
{
public:
    using IAllocator::GetAllocation;
    virtual PacBio::Memory::SmartAllocation GetAllocation(size_t count, const AllocationMarker& marker) = 0;
    virtual SmartDeviceAllocation GetDeviceAllocation(size_t count, const AllocationMarker& marker) = 0;

};

enum class CachingMode
{
    ENABLED,
    DISABLED
};

// Supported flavors of host allocations for mongo
enum class AllocatorMode
{
    MALLOC,
    CUDA,
    HUGE_CUDA,
    SHARED_MEMORY_HUGE_CUDA
};

// Sets the global allocation mode.  This is controlled
// on an application level, to better faciliate scenarios
// such as running a host-only version on a computer without
// a GPU (where allocating pinned host memory would fail)
void SetGlobalAllocationMode(CachingMode caching, AllocatorMode alloc);

// Gets globabl IMongoCachedAllocator, using whatever mode
// the application as a whole has been configured to use
IMongoCachedAllocator& GetGlobalAllocator();

// Creates an instance of IMongoCachedAllocator of a given mode,
// regardless of how the whole application is configured.
// Note: The caching for each mode is done statically and is
//       shared between instances of each mode.  This function
//       will allow you to diverge from the globally set mode,
//       but it will not give you a separate and indpendant
//       allocation cache
std::unique_ptr<IMongoCachedAllocator> CreateAllocator(
    AllocatorMode alloc,
    const AllocationMarker& marker);

void ReportAllMemoryStats();

// Toggles caching for the individual allocator modes
void EnableHostCaching(AllocatorMode mode);
void EnableGpuCaching();
void DisableHostCaching(AllocatorMode mode);
void DisableGpuCaching();
void DisableAllCaching();


}}} // ::PacBio::Cuda::Memory

#endif // PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS
