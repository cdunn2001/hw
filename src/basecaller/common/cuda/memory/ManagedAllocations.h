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

#ifndef PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS
#define PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS

#include <mongo-config.h>

#include <common/cuda/memory/SmartHostAllocation.h>
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

// These functions define the memory management API.  One might be tempted to bundle them together as
// static functions in a class.  However there is obviously state associated with these routines,
// and having static member variables opens us up to initialization ordering issues.  Instead these
// are left as free functions, with the associated state handled by a singleton class in the
// implementation file.

SmartHostAllocation GetManagedHostAllocation(size_t size, const AllocationMarker& marker);
SmartDeviceAllocation GetManagedDeviceAllocation(size_t size, const AllocationMarker& marker);

void ReturnManagedHostAllocation(SmartHostAllocation alloc);
void ReturnManagedDeviceAllocation(SmartDeviceAllocation alloc);

// Enable performance mode to use pinned memory on the host
// (which is necessary for efficient data transfers) and re-use
// memory allocations when possible (which is necessary as cuda
// malloc functions seem surprisingly slow).  The only reason this
// is configurable is to support limited use of this software on
// machines that may not have gpu hardware
void EnablePerformanceMode();
void DisablePerformanceMode();

}}} // ::PacBio::Cuda::Memory

#endif // PACBIO_CUDA_MEMORY_MANAGED_ALLOCATIONS
