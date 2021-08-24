// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#ifndef PACBIO_BAZIO_WRITING_GROWABLE_ARRAY_H
#define PACBIO_BAZIO_WRITING_GROWABLE_ARRAY_H

#include <pacbio/memory/IAllocator.h>
#include <pacbio/PBException.h>

namespace PacBio {
namespace BazIO {

/// The primary purpose of this class is to allow us to
/// coallesce a lot of tiny memory snippets into larger
/// contiguous allocations.  We could almost use std::vector
/// or std::deque for this purpose, but we want to plug into
/// the IAllocator framework, so that we can integrate well
/// with things like the allocation caching and monitoring
/// framework used in Kestrel.
///
/// The class presents itself as a 1D array of T, unless you
/// provide the optional `allocLen` paramter and it's closer
/// to a 1D array of t[allocLen].  Adding additional entires to
/// this "array" is guaranteed to not invalidate any pointers
/// or references to existing elements.
///
///
/// This is intended for cases where individual memory snippets are
/// so small  that explicitly storing a pointer to each snippet
/// would noticably impact the overal memory consumption.  As
/// such, `allocGranularity` should be large enough that a
/// pointer to each allocation is negligable, but small enough
/// that having one extra and only lightly used allocation at the
/// end does not have a huge impact on our total memory consumption.
///
/// Note: New entries into this array are default (not value) initialized.
template <typename T>
class GrowableArray
{
public:
    /// \param allocator The IAllocator instance for provisioning actual allocations
    /// \param allocGranularity The number of T (or T[]) to be contained in each
    ///                         individual allocation
    /// \param allocLen Optional parameter, if you'd like to use this class as
    ///                 a 2D T[allocLen][] array.  Defaults to 1.
    GrowableArray(std::shared_ptr<Memory::IAllocator> allocator,
                  size_t allocGranularity,
                  size_t allocLen = 1)
        : allocGranularity_(allocGranularity)
        , allocLen_(allocLen)
        , size_(0)
        , allocator_(allocator)
    {
        if (allocGranularity == 0)
            throw PBException("AllocGranularity must be larger than 0");
        if (allocLen == 0)
            throw PBException("allocLen must be larger than 0)");
    }

    GrowableArray(const GrowableArray&) = delete;
    GrowableArray(GrowableArray&&) = default;
    GrowableArray& operator=(const GrowableArray&) = delete;
    GrowableArray& operator=(GrowableArray&&) = default;

    ~GrowableArray()
    {
        for (size_t i = 0; i < size_; ++i)
        {
            T* dat = (*this)[i];
            for (size_t j = 0; j < allocLen_; ++j)
            {
                dat[j].~T();
            }
        }
    }

    /// Makes sure enough storage for count entries exists.  Has no effect if
    /// request is already satisfied
    ///
    /// \param count Requested number of array elements to ensure storage for.
    void Reserve(size_t count)
    {
        size_t numAllocs = (count + allocGranularity_-1) / allocGranularity_;
        while (allocations_.size() < numAllocs)
        {
            allocations_.emplace_back(allocator_->GetAllocation(allocGranularity_*allocLen_*sizeof(T)));
        }
    }

    /// Resizes the class up (if necessary) to be at least count elements
    /// long.  Has no effect if we are already at least this large.
    ///
    /// \param count Desired minimum size for the object.  Any entries
    ///              between the current and new sizes will be default
    ///              constructed
    void GrowToSize(size_t count)
    {
        if (count <= size_) return;
        Reserve(count);
        for ( ;size_ < count; ++size_)
        {
            T* dat = (*this)[size_];
            for (size_t i = 0; i < allocLen_; ++i)
            {
                // N.B. Lack of braces is intentional!  T{} or T()
                //      will zero initialize primives and other
                //      trivially default constructible objects,
                //      and this class does not want that.
                new (dat + i) T;
            }
        }
    }

    // Returns a pointer to a new entry added to the end.
    T* AppendOne()
    {
        GrowToSize(size_+1);
        return (*this)[size_-1];
    }

    const T* operator[](size_t idx) const
    {
        assert(idx / allocGranularity_ < allocations_.size());
        return allocations_[idx / allocGranularity_].get<T>()
            + (idx % allocGranularity_) * allocLen_;
    }

    T* operator[](size_t idx)
    {
        return const_cast<T*>(const_cast<const GrowableArray<T>&>(*this)[idx]);
    }

    size_t Size() const { return size_; }
    size_t ElementLen() const { return allocLen_; }
private:
    size_t allocGranularity_;
    size_t allocLen_;

    // Helper class to better handle moved-from semantics.  Everything but the
    // size parameter had a sensible "empty" moved from state, so it was either
    // write a wrapper class to set this value to 0 after a move, or write
    // custom move operators for the outer class that would be more involved,
    // and have to be updated any time a new member was added or removed.
    struct Size {
        Size(size_t s)
            : size(s)
        {}
        Size(const Size&) = default;
        Size(Size&& s)
        {
            size = s.size;
            s.size = 0;
        }
        Size& operator=(const Size&) = default;
        Size& operator=(Size&& s)
        {
            size = s.size;
            s.size = 0;
            return *this;
        }
        operator size_t&() { return size; }
        operator const size_t&() const { return size; }
    private:
        size_t size;
    } size_;

    std::shared_ptr<Memory::IAllocator> allocator_;
    std::vector<Memory::SmartAllocation> allocations_;
};

}}

#endif //PACBIO_BAZIO_WRITING_GROWABLE_ARRAY_H
