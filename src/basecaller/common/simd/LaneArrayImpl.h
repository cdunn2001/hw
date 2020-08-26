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
//
// \file
//  This file contains specializations of LaneArray for all supported types.
//  This includes float, int32_t, uint32_t, int16_t and uint16_t.  These
//  specializations are the insertion point for any type specific functionality,
//  e.g. `isnan` or `exp` for floats.

#ifndef mongo_common_simd_LaneArrayImpl_H_
#define mongo_common_simd_LaneArrayImpl_H_

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <array>

#include <common/MongoConstants.h>

#include <common/LaneArray_fwd.h>
#include <common/simd/ArithmeticArray.h>
#include <common/simd/BaseArray.h>
#include <common/simd/LaneArrayTraits.h>
#include <common/simd/LaneMaskImpl.h>

namespace PacBio {
namespace Simd {

// Tiny helper to handle the scalar to vector
// conversions that need to happen when
// a LaneArray declares it's parent class type
template <typename T, size_t ScalarCount,
          template <typename, size_t> class Child>
using ArithmeticBase = ArithmeticArray<vec_type_t<T>,
                                       ScalarCount / vec_type_t<T>::size(),
                                       Child<T, ScalarCount>>;

template <size_t ScalarCount>
class LaneArray<float, ScalarCount> : public ArithmeticBase<float, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<float, ScalarCount, LaneArray>;
public:
    using Base::Base;

    friend LaneMask<ScalarCount> isnan(const LaneArray& c)
    {
        return LaneMask<ScalarCount>(
            [](auto&& d){ return isnan(d); },
            c);
    }

    friend LaneMask<ScalarCount> isfinite(const LaneArray& c)
    {
        return LaneMask<ScalarCount>(
            [](auto&& d){ return isfinite(d); },
            c);
    }

    friend LaneArray erfc(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return erfc(in2); },
            in);
    }

    friend LaneArray log(const LaneArray<float, ScalarCount>& in)
    {
        return LaneArray(
            [](auto&& in2){ return log(in2); },
            in);
    }

    friend LaneArray log2(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return log2(in2); },
            in);
    }

    friend LaneArray exp(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return exp(in2); },
            in);
    }

    friend LaneArray exp2(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return exp2(in2); },
            in);
    }

    friend LaneArray sqrt(const LaneArray& in)
    {
        return LaneArray(
            [](auto&& in2){ return sqrt(in2); },
            in);
    }

    friend LaneArray<int, ScalarCount> floorCastInt(const LaneArray& in)
    {
        return LaneArray<int, ScalarCount>(
            [](auto&& in2){ return floorCastInt(in2); },
            in);
    }
};

template <size_t ScalarCount>
class LaneArray<uint32_t, ScalarCount> : public ArithmeticBase<uint32_t, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<uint32_t, ScalarCount, LaneArray>;
public:
    using Base::Base;
};

template <size_t ScalarCount>
class LaneArray<uint16_t, ScalarCount> : public ArithmeticBase<uint16_t, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<uint16_t, ScalarCount, LaneArray>;
public:
    using Base::Base;
};

template <size_t ScalarCount>
class LaneArray<int32_t, ScalarCount> : public ArithmeticBase<int32_t, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<int32_t, ScalarCount, LaneArray>;
public:
    using Base::Base;

    friend LaneArray operator|(const LaneArray& l, const LaneArray& r)
    {
        return LaneArray(
            [](auto&& l2, auto&& r2) { return l2 | r2; },
            l, r);
    }

};

template <size_t ScalarCount>
class LaneArray<int16_t, ScalarCount> : public ArithmeticBase<int16_t, ScalarCount, LaneArray>
{
    using Base = ArithmeticBase<int16_t, ScalarCount, LaneArray>;
public:
    using Base::Base;
};

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_LaneArrayImpl_H_
