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

#ifndef PACBIO_BAZIO_WRITING_MEMORY_BUFFER_H_
#define PACBIO_BAZIO_WRITING_MEMORY_BUFFER_H_

#include <cassert>
#include <cstddef>
#include <vector>
#include <type_traits>

#include <pacbio/logging/Logger.h>
#include <pacbio/memory/SmartAllocation.h>
#include <pacbio/datasource/MallocAllocator.h>

namespace PacBio {
namespace BazIO {

template <typename T>
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
    template <typename U>
    friend class MemoryBuffer;
    MemoryBufferView(T* data, size_t size) : data_(data), size_(size) {}
public:
    MemoryBufferView() : data_(nullptr), size_(0) {}

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
// Some historical notes:
//
// The file is based off the Sequel version by the same name, in the Sequel/common
// library.  The code has been updated/simplified somewhat as the use cases are now
// somewhat different.  The old sequel version can be more-or-less considered
// mothballed, however there *also* exists a very similar AllocationSlicer code
// that exists in the pa-common respository.  (That code was also forked from the
// same original implementation).  The functionality here and there is *very*
// similar, however it's not clear that the two use cases truly align in a
// fundamental sense.  I've leaving the near-duplication of code for now, we
// can always go back and unify things later if it seems desirable.
template <typename T>
class MemoryBuffer
{
    // Private class, to handle the actual allocations and object lifetimes.
    class MemoryInnerBuffer
    {
    public:
        MemoryInnerBuffer(Memory::IAllocator& alloc, size_t capacity)
              : size_(0)
              , data_(alloc.GetAllocation(capacity*sizeof(T)))
        {}

        MemoryInnerBuffer(const MemoryInnerBuffer&) = delete;
        MemoryInnerBuffer(MemoryInnerBuffer&&) = default;
        MemoryInnerBuffer& operator=(const MemoryInnerBuffer&) = delete;
        MemoryInnerBuffer& operator=(MemoryInnerBuffer&&) = default;

        ~MemoryInnerBuffer()
        {
            ResetInner();
        }

        /// Does not release any memory, but will destroy any objects that
        /// have been created, effectively turning things back into uninitialized
        /// allocations
        void ResetInner()
        {
            static_assert(std::is_nothrow_destructible<T>::value, "MemoryBuffer requires a nothrow destructor!");

            if (data_)
            {
                auto ptr = data_.get<T>();
                // Really shouldn't have to do this optimization manually,
                // but since I literally saw another piece of code fail to
                // optimize away a loop over no-op destructors, I'm
                // officially paranoid about such things.
                if (!std::is_trivially_destructible<T>::value)
                {
                    for (size_t i = 0; i < size_; ++i)
                    {
                        ptr[i].~T();
                    }
                }
            }
            size_ = 0;
        }
        /// \returns number of T objects in this buffer
        size_t size() const { return size_; }

        /// \returns number of T objects that this buffer could potentially hold without realllocation
        size_t capacity() const { return data_.size() / sizeof(T); }

        /// Convenience function, to see if there is space enough for n more items
        bool CanHold(size_t n) const { return size_ + n <= capacity(); }

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
            if (size_ + n > capacity())
                throw PBException("Bounds overrun in MemoryBuffer");
            auto ret = MemoryBufferView<T>(data_.get<T>() + size_, n);
            size_ += n;

            return ret;
        }

        size_t size_;
        Memory::SmartAllocation data_;
    };

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
    MemoryBuffer(size_t bufferSize, size_t initialBufferCount,
                 std::shared_ptr<Memory::IAllocator> allocator)
      : size_(0)
      , bufferSize_(bufferSize)
      , bufferIdx_(0)
      , allocator_(allocator)
    {
        for (size_t i = 0; i < initialBufferCount; ++i)
        {
            storage_.emplace_back(*allocator_, bufferSize_);
        }
    }

    MemoryBuffer(const MemoryBuffer&) = delete;
    MemoryBuffer(MemoryBuffer&&) = default;
    MemoryBuffer& operator=(const MemoryBuffer&) = delete;
    MemoryBuffer& operator=(MemoryBuffer&&) = default;

    ~MemoryBuffer()
    {
        Reset();
    }

    void Reset()
    {
        for (auto& data : storage_) data.ResetInner();

        size_ = 0;
        bufferIdx_ = 0;
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

    size_t Capacity() const { return bufferSize_ * storage_.size(); }
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
    MemoryInnerBuffer& Prepare(size_t n)
    {
        if (n > bufferSize_)
            throw PBException("MemoryBuffer allocation request size too large");

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
                try
                {
                    storage_.emplace_back(*allocator_, bufferSize_);
                }
                catch(...)
                {
                    PBLOG_NOTICE    << " MemoryBuffer::Prepare this:" << (void*) this
                                    << " n:" << n << " size_:" << size_;
                    PBLOG_ERROR << "Exception caught while allocating another storage buffer";
                    throw;
                }
                bufferIdx_ = storage_.size()-1;

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
    size_t bufferSize_;         // size (in bytes) of the underlying backend buffers
                                // we will use
    size_t bufferIdx_;          // Idx of the buffer we're currently filling
    std::shared_ptr<Memory::IAllocator> allocator_;    // Helper class to actually allocate memory
    std::vector<MemoryInnerBuffer> storage_;  // All memory that has been allocated
};

}} // ::PacBio::Primary

#endif // PACBIO_BAZIO_WRITING_MEMORY_BUFFER_H_
