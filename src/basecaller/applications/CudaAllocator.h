// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_APPLICATION_CUDA_ALLOCATOR_H
#define PACBIO_APPLICATION_CUDA_ALLOCATOR_H

#include <pacbio/logging/Logger.h>
#include <pacbio/memory/IAllocator.h>
#include <pacbio/memory/SmartAllocation.h>
#include <pacbio/PBException.h>

#include <common/cuda/memory/ManagedAllocations.h>

namespace PacBio {
namespace Application {

// There is an unfortunate quirk when using this class, where in part behaviour
// is coupled to a global system.
//
// Once upon a time, each object that may want to allocate cuda aware memory had
// it's own allocation pool, which could be independantly set to use cuda memory
// or not, and this was necessary to enable certain tests to run without dragging
// in the whole cuda runtime as a dependancy.
//
// However at some point it got unwieldy having a myriad of independant allocation
// pools, both because they were a pain to set up, and because it prevented efficient
// allocation sharing and good usage diagnostics/statistics.  It was also decided
// that it was undesirable to have a non-global allocation pool set up that explicitly
// threads all of our call stacks.  So ManagedAllocations.h was set up as a global
// memory allocator of sorts, with a few global functions as access points.
//
// Ideally, the EnablePerformanceMode() function is called before any instances of
// this class are created, and DisablePerformanceMode() is called after they are all
// destroyed.  This allocator class will work correctly regardless, in that memory
// leaks should be impossible and any allocated memory will have the desired characteristics,
// but only using this while "Performance Mode" is active will make sure allocations are
// cached and re-used, and correct bookkeeping is done to track allocation statistics
class CudaAllocator : public Memory::IAllocator
{
public:
    CudaAllocator()
    {
        if(Cuda::Memory::GetCurrentAllocType() != Cuda::Memory::ManagedAllocType::PINNED)
        {
            PBLOG_WARN << "Creating a CudaAllocator while pinned memory is disabled by default.  "
                       << "Allocations will succeed correctly, but they may not be cached and "
                       << "usage statistics may be missing incorrect";
        }
    }
    Memory::SmartAllocation GetAllocation(size_t count) override
    {
        return GetManagedHostAllocation(count, SOURCE_MARKER(), Cuda::Memory::ManagedAllocType::PINNED);
    }

    bool SupportsAllFlags(uint32_t flags) const override
    {
        if (flags | IAllocator::HUGEPAGES) throw PBException("Huge pages not supported by this Allocator");

        return true;
    }
};

}}

#endif
