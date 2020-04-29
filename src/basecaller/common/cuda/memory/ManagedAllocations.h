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

// Ideally external code will not specify the ManagedAllocType
// (effectively using DEFAULT) so that there can be a cohesive
// allocation strategy.  However, this is a global system, and
// to allow it to interact with an RAII class like an instance
// of an IAllocator, one needs to be able to request a specific
// allocation type to satisfy whatever requirements are
// associated with that instance.  Efficient allocations and
// correct statistics will still require the global state and
// RAII instances to remain in coordination with each other, but
// doing things this way allows us to maintain correctness
// in terms of the actual properties of individual allocations,
// as well as detect and warn if the caching/statistics are
// not going to work properly.
class ManagedAllocType
{
public:
    enum Flags : uint32_t
    {
        DEFAULT  = 0,
        STANDARD = 1>>0,
        PINNED   = 1>>1,
        HUGEPAGE = 1>>2  // Not yet supported
    };

    ManagedAllocType(uint32_t flags)
        : val_(flags)
    {
        if ((flags & STANDARD) && (flags != STANDARD))
            throw PBException("STANDARD allocation flag incompatible with other flags");
    }

    bool operator==(uint32_t flags) const
    {
        return val_ == flags;
    }

    bool operator!=(uint32_t flags) const
    {
        return !(*this == flags);
    }

    bool Supports(uint32_t flags) const
    {
        return (val_ & flags) == flags;
    }

    uint32_t Flags() const { return val_; }
private:
    uint32_t val_;
};


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
// Automatically strips out portions of the path outside of the repository
#define SOURCE_MARKER()                                                        \
        PacBio::Cuda::Memory::AllocationMarker(                                \
                std::string(__FILE__  ":" + std::to_string(__LINE__))          \
                .substr(strlen(PacBio::Primary::MongoConfig::workspaceDir)+1))

// Extension of the IAllocator interface, which provides two new piece
// of functionality:
// 1. Allows cuda allocations (which are a fundamentally different type from
//    host allocations)
// 2. Facilitates allocation caching, which in some situations can be significantly
//    faster than allocating/deallocating individual allocations each time they are
//    needed
class IMongoCachedAllocator : public PacBio::Memory::IAllocator
{
public:
    using IAllocator::GetAllocation;
    virtual PacBio::Memory::SmartAllocation GetAllocation(size_t count, const AllocationMarker& marker) = 0;
    virtual SmartDeviceAllocation GetDeviceAllocation(size_t count, const AllocationMarker& marker) = 0;

    // Opt-in routines that allow allocation caching.  If you let a SmartAllocation expire it will
    // be deallocated, but if instead you return it via one of these functions, it may be recycled
    // to satisfy future allocation requests.
    // Note: Allocation caching can be disabled application wide.  If allocation is disabled,
    //       then calling these functions will simply free the memory
    // Note: Different instances of IMongoCachedAllocator may share the same allocation cache.
    //       Certain things cannot be mixed (like huge pages and normal system allocations),
    //       but otherwise the implementation will share allocation pools between instances
    // Note: The allocation caching scheme is currently very simple.  Calling these "Return"
    //       functions for frequent one-off allocations of irregular size will probably result
    //       in them never being recycled, and they will live until the cache is manually
    //       cleared (probably not until the end of execution).  In this situation it will
    //       effectively appear to be a memory leak.
    void ReturnHostAllocation(PacBio::Memory::SmartAllocation alloc);
    void ReturnDeviceAllocation(SmartDeviceAllocation alloc);
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
    HUGE_CUDA
};

// Sets the global allocation mode.  This is controlled
// on an application level, to better faciliate scenarious
// such as running a host-only version on a computer without
// a GPU (where allocating pinned host memory would failed)
void SetGlobalAllocationMode(CachingMode caching, AllocatorMode alloc);

// Gets globablly IMongoCachedAllocator, using whatever mode
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

// Toggles caching for the individual allocator modes
void EnableHostCaching(AllocatorMode mode);
void EnableGpuCaching();
void DisableHostCaching(AllocatorMode mode);
void DisableGpuCaching();
void DisableAllCaching();


}}} // ::PacBio::Cuda::Memory

#endif // PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS
