#ifndef Sequel_Common_MemoryBuffer_H_
#define Sequel_Common_MemoryBuffer_H_

// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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
//
//  Description:
/// \file   MemoryBuffer.h
/// \brief  This class is used to help coalesce memory allocations, reserving
///         memory from the system using a few large allocations, and allowing
///         client code to consume that with a large number of smaller requests

#include <cassert>
#include <cstddef>
#include <vector>
#include <type_traits>

#ifdef SECONDARY_BUILD
#include "pacbio/microlog/Logger.h"
using namespace PacBio::MicroLog;
#else
#include <pacbio/logging/Logger.h>
#endif

namespace PacBio {
namespace Primary {

template <typename T, template <typename U> class Allocator>
class MemoryBuffer;

// This class is used to wrap a pointer belonging to the MemoryBuffer.  It is a
// NON-OWNING object, and the lifetime of the memory is determined by the lifetime
// of the MemoryBuffer, not any instance of this class.  It is recommended that
// you carefully control the propagation MemoryBufferView objects in order to
// prevent dangling references.
template <typename T>
class MemoryBufferView
{
    // Private CTor, a MemoryBuffer is the only thing capable of creating
    // a "non-null" view object.
    template <typename U, template <typename V> class Allocator>
    friend class MemoryBuffer;
    MemoryBufferView(T* data, size_t size) : data_(data), size_(size) {}
public:
    MemoryBufferView() : data_(nullptr), size_(0) {}

    MemoryBufferView(const MemoryBufferView&) = default;
    MemoryBufferView(MemoryBufferView&&) = default;
    MemoryBufferView& operator=(const MemoryBufferView&) = default;
    MemoryBufferView& operator=(MemoryBufferView&&) = default;

    // Very basic array/container interface.
    bool empty() const { return size_ == 0; }
    size_t size() const {return size_;}
    T& operator[] (size_t idx) { assert(idx < size_); return data_[idx]; }
    const T& operator[] (size_t idx) const { assert(idx < size_); return data_[idx]; }
    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    T* data_;
    size_t size_;
};

// Buffer class used to coalesce memory allocations.  It makes a few large
// allocations from the system, and parceling that out to client code over
// (potentially) many small allocations.  This eases the burden on the system
// allocator when you have a lot of tiny unpredictable allocations to make.
//
// This class guarantees that any allocation will give contiguous memory large
// enough to hold the requested number of elements (assuming the system itself
// can provide that much).  It also guarantees that any previous allocations
// will remain valid for the lifetime of the buffer itself. Causing the class
// to add more capacity will not cause previous allocations to be moved in memory.
//
// Allocator template parameter is mostly meant as a hook for testing code
template <typename T, template <typename U> class Allocator = std::allocator>
class MemoryBuffer
{
    // Private class, to handle the actual allocations and object lifetimes.
    class MemorInnerBuffer
    {
    public:
        MemorInnerBuffer(const MemorInnerBuffer&) = delete;
        MemorInnerBuffer& operator=(const MemorInnerBuffer&) = delete;

        MemorInnerBuffer(MemorInnerBuffer&& other)
              : size_(other.size_),
                capacity_(other.capacity_),
                alloc_(other.alloc_),
                data_(other.data_)
        {
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        MemorInnerBuffer& operator=(MemorInnerBuffer&& other)
        {
            if (data_ != nullptr) alloc_->deallocate(data_, capacity_);

            alloc_ = other.alloc_;
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;

            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
            return *this;
        }

        MemorInnerBuffer(Allocator<T>&  alloc, size_t capacity)
              : size_(0)
              , capacity_(capacity)
              , alloc_(&alloc)
              , data_(alloc.allocate(capacity))
        {

        }

        ~MemorInnerBuffer()
        {
            ResetInner();
            if (data_) alloc_->deallocate(data_, capacity_);
        }

        /// Does not release any memory, but will destroy any objects that
        /// have been created, effectively turning things back into uninitialized
        /// allocations
        void ResetInner()
        {
            //static_assert(std::is_nothrow_destructible<T>::value, "MemoryBuffer requires a nothrow destructor!");

            // stopgap guard against double destruction if one of these destructors
            // throws and then the stack unwinds and calls this function again.
            // Things are still fubar and we may have resource leaks, but at least
            // we wont call std::terminate because of two active exceptions.
            // Preferably we'd just disable throwing destructors, but the
            // current mic compiler doesn't recognize std::is_nothrow_destructible
            // Can maybe revisit after that is upgraded.
            const auto size = size_;
            size_ = 0;
            if (data_)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    data_[i].~T();
                }
            }
        }
        /// \returns number of T objects in this buffer
        size_t size() const { return size_; }

        /// \returns number of T objects that this buffer could potentially hold without realllocation
        size_t capacity() const { return capacity_; }

        /// Convenience function, to see if there is space enough for n more items
        bool CanHold(size_t n) const { return size_ + n <= capacity_; }

        /// Returns a MemoryBufferView with a copy of data.  Written such that
        /// if T is a trivially copiable type, this should just optimize down
        /// to a bulk memcpy or equivalent.
        MemoryBufferView<T> Copy(const T* data, size_t n)
        {
            auto ret = Prepare(n);
            std::uninitialized_copy(data, data+n, ret.data());
            return ret;
        }

        /// Returns a MemoryBufferView containing n default-initialized elements
        /// of T.  Default initialization is an intentional choice, so that it
        /// completely optimizes away for simple types. (wont even memset to 0)
        MemoryBufferView<T> Allocate(size_t n)
        {
            auto ret = Prepare(n);
            for (size_t i = 0; i < n; ++i)
            {
                // Lack of () or {} on T is very intentional!  Necessary to
                // invoke default initialization.  Complicated types will get
                // their default constructor invoked regardless, and doing
                // value initialization on very simple types will result in an
                // unecessary zeroing of the memory.
                new (ret.data()+i) T;
            }
            return ret;
        }

        /// Returns a MemoryBufferView containing a single newly constructed
        /// T value
        template <class... Args>
        MemoryBufferView<T> Construct(Args&&... args)
        {
            auto ret = Prepare(1);
            new (ret.data()) T(std::forward<Args>(args)...);
            return ret;
        }

    private:
        /// Creates a MemoryBufferView for the next n elements, and updates
        /// internal counters accordingly
        MemoryBufferView<T> Prepare(size_t n)
        {
            if (size_ + n > capacity_)
                throw PBException("Bounds overrun in MemoryBuffer");
            auto ret = MemoryBufferView<T>(data_ + size_, n);
            size_ += n;

            return ret;
        }

        size_t size_;
        size_t capacity_;
        Allocator<T>* alloc_;
        T* data_;
    };

    // fast (linear) growth rate = fastRatio * estimatedMaximum;
    static constexpr float fastRatio = .25;
    // slow (exponential) growth rate = slowRatio * size_;
    static constexpr float slowRatio = .1;

public:
    // initialCapacity: The initial allocation that will be made upon creation
    // estimatedMaximum: The predicted final usage.  This is used to determine
    // growth strategy whenever additional memory needs to be reserved.
    // Before reaching the maximum it will do a fast linear growth, in order to
    // do as few allocations as possible.  Past the maximum it will have a
    // slower (but exponential) growth in order to remain flexible but not
    // over-allocate by a very large margin.
    //
    // It is recommended that the `estimatedMaximum` be as accurate as possible,
    // to avoid either an excessive number of allocations, or an excessive
    // amount of over-allocations.  If this proves impossible it may prove
    // worthwhile to make the growth rates configurable.
    MemoryBuffer(size_t initialCapacity, size_t estimatedMaximum)
      : size_(0)
      , capacity_(initialCapacity)
      , estimatedMaximum_(estimatedMaximum)
      , fastAllocations_(0)
      , slowAllocations_(0)
      , bufferIdx_(0)
      , allocator_()
    {
        storage_.emplace_back(allocator_, capacity_);
        PBLOG_DEBUG << " MemoryBuffer::Constructor2 this:" << (void*) this
                    << " capacity_:" << capacity_
                    << " estimatedMaximum_:" << estimatedMaximum_;
    }

    // Single argument constructor, if you know exactly how much memory you
    // want and wish to skip the initial growth phase.
    MemoryBuffer(size_t capacity) : MemoryBuffer(capacity, capacity)
    {
        PBLOG_DEBUG << " MemoryBuffer::Constructor1 this:" << (void*) this
                    << " capacity_:" << capacity_;
    }

    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer(MemoryBuffer&&) = delete;
    MemoryBuffer& operator=(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(MemoryBuffer&&) = delete;

    ~MemoryBuffer()
    {
        Reset();
    }

    void Reset()
    {
        for (auto& data : storage_) data.ResetInner();

        size_ = 0;
        bufferIdx_ = 0;

        fastAllocations_ = 0;
        slowAllocations_ = 0;
    }

    // Copies the data into the buffer and returns a MemoryBufferView to the
    // new copy
    MemoryBufferView<T> Copy(const T* data, size_t n)
    {
        return Prepare(n).Copy(data, n);
    }

    // Allocates space for n elements and returns a MemoryBufferView.  Requires
    // the underlying type T to be default constructible.
    //
    // Note: if you need T to be non default constructible but do not wish to
    // call this function, you can hide it behind a dummy template parameter
    // so that it never gets instantiated for that particular type.
    MemoryBufferView<T> Allocate(size_t n)
    {
        return Prepare(n).Allocate(n);
    }

    // Forwarding constructor to create a single instance of type T.
    template<typename... Args>
    MemoryBufferView<T> Create(Args&&... args)
    {
        return Prepare(1).Construct(std::forward<Args>(args)...);
    }

    size_t Capacity() const { return capacity_; }
    size_t Size() const { return size_; }

    // Helper method to determine how much memory waste there is.  Due to the
    // guarantees this class makes about not invalidating previous pointers,
    // any allocation that requires a capacity increase will likely leave
    // behind some unused slots in the previous segment of memory.  If any
    // allocations are very large this waste could potentially be large as well.
    float FragmentationFraction() const
    {
        if (size_ == 0) return 0;

        double waste = 0;
        for (const auto& buffer : storage_)
            waste += buffer.capacity() - buffer.size();
        // The buffer we're actively filling shouldn't count as waste.
        waste -= storage_[bufferIdx_].capacity() - storage_[bufferIdx_].size();
        return static_cast<float>(waste / (size_));
    }

private:

    // Function to handle all the internal storage buffers. It is guaranteed
    // to return a reference to a MemoryPool that can hold at least n more
    // elements, by either finding an existing one with sufficient capacity,
    // or creating a new one that will allocate a fresh batch of memory
    MemorInnerBuffer& Prepare(size_t n)
    {
        if (!storage_[bufferIdx_].CanHold(n))
        {
            bool found = false;
            for (size_t i = 0; i < storage_.size(); ++i)
            {
                if (storage_[i].CanHold(n))
                {
                    bufferIdx_ = i;
                    found = true;
                    break;
                }
            }

            if(!found)
            {
                size_t nextGrowth = 0;
                size_t fastGrowth = fastRatio * estimatedMaximum_;
                size_t slowGrowth = slowRatio * size_;
                if (size_ + fastGrowth < estimatedMaximum_)
                {
                    nextGrowth = fastGrowth;
                    fastAllocations_++;
                }
                // Try not to overshoot the estimatedMaximum too badly
                else if (size_ + fastGrowth - estimatedMaximum_ < 0.25 * fastGrowth)
                {
                    nextGrowth = fastGrowth;
                    fastAllocations_++;
                }
                else
                {
                    nextGrowth = slowGrowth;
                    slowAllocations_++;
                }
                if (nextGrowth < n) nextGrowth = n;

                try
                {
                    PBLOG_DEBUG << " MemoryBuffer::Prepare this:" << (void*) this
                                << " n:" << n << " slowAllocations_:" << slowAllocations_
                                << " fastAllocations_:" << fastAllocations_
                                << " fastGrowth:" << fastGrowth << " slowGrowth:" << slowGrowth
                                << " fastRatio:" << fastRatio << " estimatedMaximum_:" << estimatedMaximum_
                                << " size_:" << size_ << " nextGrowth:" << nextGrowth;
                    storage_.emplace_back(allocator_, nextGrowth);
                }
                catch(...)
                {
                    PBLOG_NOTICE    << " MemoryBuffer::Prepare this:" << (void*) this
                                    << " n:" << n << " slowAllocations_:" << slowAllocations_
                                    << " fastAllocations_:" << fastAllocations_
                                    << " fastGrowth:" << fastGrowth << " slowGrowth:" << slowGrowth
                                    << " fastRatio:" << fastRatio << " estimatedMaximum_:" << estimatedMaximum_
                                    << " size_:" << size_ << " nextGrowth:" << nextGrowth;
                    PBLOG_ERROR << "Exception caught during storage_.emplace_back(allocator_.maxSize:"
                                << allocator_.max_size()
                                << ", nextGrowth=" << nextGrowth << ")";
                    throw;
                }
                capacity_ += storage_.back().capacity();
                bufferIdx_ = storage_.size()-1;

                if (slowAllocations_ > 100)
                    // Should fall back to standard 1.5x growth or something in this case?  Obviously provided estimates are bad...
                    PBLOG_WARN << "Excessive small allocations in MemoryBuffer: " << slowAllocations_;
                if (FragmentationFraction() > fragmentationWarningThresh && size_ * sizeof(T) > 1000000)
                {
                    fragmentationWarningThresh += .1;
                    PBLOG_WARN << "Excessive memory fragmentation in MemoryBuffer: " << FragmentationFraction();
                }
            }
        }

        size_ += n;
        assert(storage_[bufferIdx_].CanHold(n));
        return storage_[bufferIdx_];
    }

    float fragmentationWarningThresh = 0.1;

    size_t size_;               // elements currently allocated and initialized
    size_t capacity_;           // number of elements we've reserved space for
    size_t estimatedMaximum_;   // Guessed final number of elements needed
    size_t fastAllocations_;    // Number of "fast" (linear growth) allocations done
    size_t slowAllocations_;    // Number of "slow" (exponential growth) allocations done

    size_t bufferIdx_;          // Idx of the buffer we're currently filling
    Allocator<T> allocator_;    // Helper class to actually allocate memory
    std::vector<MemorInnerBuffer> storage_;  // All memory that has been allocated
};

}} // ::PacBio::Primary

#endif // Sequel_Common_MemoryBuffer_H_
