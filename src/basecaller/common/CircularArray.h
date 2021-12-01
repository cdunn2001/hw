#ifndef mongo_common_CircularArray_H_
#define mongo_common_CircularArray_H_

// Copyright (c) 2015-2021, Pacific Biosciences of California, Inc.
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
/// \file  CircularArray.h
/// \brief A fixed-size circular buffer with compile-time defined capacity.

#include <array>
#include <cassert>

namespace PacBio::Mongo {

/// A simple fixed-size circular_buffer with capacity defined at compile time.
/// Requires no dynamic memory allocation.
template <typename T, size_t Capacity>
class CircularArray
{
public:
    using value_type = T;

public:
    CircularArray() : i0_(0), i1_(-1)
    { }

    void push_back(const T& item)
    {
        if (full()) i0_ = (i0_ + 1) % Capacity;

        i1_ = (i1_ + 1) % Capacity;  // OK for size 0 (i1_ = -1) as well
        buf_[i1_] = item;
    }

    void pop_back()
    {
        assert(!empty());
        if (size() == 1)
        {   // reset to empty state
            i0_ = 0;
            i1_ = -1;
        }
        else
        {
            i1_ = (Capacity + i1_ - 1) % Capacity;
        }
    }

    void push_front(const T& item)
    {
        if (empty())
        {
            assert(i0_ == 0);
            buf_[i0_] = item;
            i1_ = 0;
        }
        else
        {
            bool isfull = full();
            i0_ = (Capacity + i0_ - 1) % Capacity;
            if (isfull)
                i1_ = (i0_ + Capacity - 1) % Capacity;

            buf_[i0_] = item;
        }
    }

    void pop_front()
    {
        assert(!empty());

        if (size() == 1)
        {   // reset to empty state
            i0_ = 0;
            i1_ = -1;
        }
        else
        {
            i0_ = (i0_ + 1) % Capacity;
        }
    }

    T& operator[](size_t i)
    {
        assert(i < size());
        return buf_[(i0_ + i) % Capacity];
    }

    const T& operator[](size_t i) const
    {
        assert(i < size());
        return buf_[(i0_ + i) % Capacity];
    }

    const T& front() const
    {
        assert(size() > 0);
        return buf_[i0_];
    }

    T& front()
    {
        assert(size() > 0);
        return buf_[i0_];
    }

    const T& back() const
    {
        assert(size() > 0);
        return buf_[i1_];
    }

    T& back()
    {
        assert(size() > 0);
        return buf_[i1_];
    }

    size_t capacity() const
    { return Capacity; }

    size_t size() const
    { return i1_ < 0 ? 0 : 1 + (Capacity + i1_ - i0_) % Capacity; }

    bool empty() const
    { return size() == 0; }

    bool full() const
    { return size() == Capacity; }

    void clear()
    {
        i0_ = 0;
        i1_ = -1;
    }

private:
    std::array<T, Capacity> buf_;
    int i0_;    // index to the first element
    int i1_;    // index to the last element
};

}       // namespace PacBio::Mongo

#endif  // mongo_common_CircularArray_H_
