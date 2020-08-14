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

#ifndef mongo_common_simd_LaneMaskImpl_H_
#define mongo_common_simd_LaneMaskImpl_H_

#include <common/simd/BaseArray.h>
#include <common/simd/m512b.h>

namespace PacBio {
namespace Simd {

// Class that represents an array of bools.  Required to be an even multiple
// of 16 elements so that we can easily using the m512b type for underlying
// storage.
template<size_t ScalarCount_>
class LaneMask : public BaseArray<m512b, ScalarCount_/16, LaneMask<ScalarCount_>>
{
    using T = m512b;
    static_assert(ScalarCount_ % 16 == 0, "");
    using Base = BaseArray<T, ScalarCount_/16, LaneMask<ScalarCount_>>;
public:
    // Inherit all the constructor magic provided by BaseArray.
    using Base::Base;

    // These are already public in the base class, but
    // re-declaring them here because they are used
    // in the implementation and this avoids typing a
    // bunch of `template`/`typename` keywords
    static constexpr auto SimdCount = Base::SimdCount;
    static constexpr auto SimdWidth = Base::SimdWidth;
    using Base::Update;
    using Base::Reduce;
    using Base::data;

    LaneMask() = default;

    // BaseArray did not have any ctors/conversions with
    // CudaArray<bool, N>, since it's not bitwise compatible
    // with the m512b type.  So we write them here.
    LaneMask(const Cuda::Utility::CudaArray<bool, ScalarCount_>& arr)
    {
        for (size_t i = 0; i < SimdCount; ++i)
        {
            auto start = i * SimdWidth;
            data()[i] = m512b(arr[start+0],  arr[start+1],  arr[start+2],  arr[start+3],
                              arr[start+4],  arr[start+5],  arr[start+6],  arr[start+7],
                              arr[start+8],  arr[start+9],  arr[start+10], arr[start+11],
                              arr[start+12], arr[start+13], arr[start+14], arr[start+15]);
        }
    }

    operator Cuda::Utility::CudaArray<bool, ScalarCount_>() const
    {
        Cuda::Utility::CudaArray<bool, ScalarCount_> ret;
        for (size_t i = 0; i < ScalarCount_; ++i)
        {
            ret[i] = (*this)[i];
        }
        return ret;
    }

public: // Reduction operators
    friend bool all(const LaneMask& m)
    {
        // Reduce accepts a lambda and an initial value.
        // The remaining arguments are the values to be
        // reduced.
        return Reduce([](auto&& l, auto&& r) { l &= all(r); }, true, m);
    }

    friend bool any(const LaneMask& m)
    {
        return Reduce([](auto&& l, auto&& r) { l |= any(r); }, false, m);
    }

    friend bool none(const LaneMask& m)
    {
        return Reduce([](auto&& l, auto&& r) { l &= none(r); }, true, m);
    }

public: // Standard logical operators
    friend LaneMask operator! (const LaneMask& m)
    {
        // This contructor accepts a lamda and and arbitrary
        // list of arguments.  Each member of the new
        // array is initialized by calling the lambda on
        // each element of the input arrays.
        return LaneMask(
            [](auto&& m2){ return !m2; },
            m);
    }

    friend LaneMask operator| (const LaneMask& l, const LaneMask& r)
    {
        return LaneMask(
            [](auto&& l2, auto&& r2){ return l2 | r2; },
            l, r);
    }

    friend LaneMask operator& (const LaneMask& l, const LaneMask& r)
    {
        return LaneMask(
            [](auto&& l2, auto&& r2){ return l2 & r2; },
            l, r);
    }

    friend LaneMask operator^ (const LaneMask& l, const LaneMask& r)
    {
        return LaneMask(
            [](auto&& l2, auto&& r2){ return l2 ^ r2; },
            l, r);
    }

public: // Compound operators
    LaneMask& operator &= (const LaneMask& o)
    {
        // Very similar to how the previous LaneMask
        // constructors worked, save that instead of
        // creating a new vector, the lambda is called
        // with elements of this array as the first member
        return Update([](auto&& l, auto&& r) { l &= r; }, o);
    }

    LaneMask& operator |= (const LaneMask& o)
    {
        return Update([](auto&& l, auto&& r) { l |= r; }, o);
    }

    // Element access.  Does not provide a reference
    // for modification.  Is not meant to be used in
    // performance sensitive code.
    bool operator[](size_t idx) const
    {
        static constexpr auto width = SimdTypeTraits<T>::width;
        return this->data_[idx / width][idx%width];
    }
};

}}

#endif // mongo_common_simd_LaneMaskImpl_H_
