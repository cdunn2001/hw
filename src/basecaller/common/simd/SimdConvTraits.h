#ifndef mongo_common_simd_SimdConvTraits_H_
#define mongo_common_simd_SimdConvTraits_H_

// Copyright (c) 2016, Pacific Biosciences of California, Inc.
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
//  Defines templates do define relationships among SIMD types.


#include "ArrayUnion.h"
#include "m32s.h"
#include "m512b.h"
#include "m512f.h"
#include "m512i.h"
#include "m512s.h"
#include "m512ui.h"
#include "m512us.h"

namespace PacBio {
namespace Simd {

template <typename _V>
struct SimdConvTraits
{
    typedef float   float_conv;    // TODO: I'm assuming nothing but 32-bit float in operation.
    typedef int32_t index_conv;    // An integer index that has the same width as type V
    typedef bool    bool_conv;
    typedef m32s pixel_conv;
    typedef ArrayUnion<_V> union_conv;
};

template <>
struct SimdConvTraits<float>
{
    typedef float   float_conv;
    typedef int32_t index_conv;
    typedef bool    bool_conv;
    typedef m32s pixel_conv;
    typedef ArrayUnion<float> union_conv;
};

template<>
struct SimdConvTraits<m512b>
{
    typedef m512b bool_conv;
    typedef m512f float_conv;
    typedef m512i index_conv;
    typedef m512s pixel_conv;
    // We cannot provide ArrayUnion<m512b> because the underlying storage
    // only consumed 16 bits.  However we're still going to want to do
    // UnionConv<bool> in scalar code so we need this identity transform here to
    // help make that work.  This way we still get across the board read access
    // through the const operator[]
    typedef m512b union_conv;
};

template<>
struct SimdConvTraits<m512f>
{
    typedef m512b bool_conv;
    typedef m512f float_conv;
    typedef m512i index_conv;
    typedef m512s pixel_conv;
    typedef ArrayUnion<m512f> union_conv;
};

template<>
struct SimdConvTraits<m512i>
{
    typedef m512b bool_conv;
    typedef m512f float_conv;
    typedef m512i index_conv;
    typedef m512s pixel_conv;
    typedef ArrayUnion<m512i> union_conv;
};

template<>
struct SimdConvTraits<m512s>
{
    typedef m512b bool_conv;
    typedef m512f float_conv;
    typedef m512i index_conv;
    typedef m512s pixel_conv;
    typedef ArrayUnion<m512s> union_conv;
};

template<>
struct SimdConvTraits<m512ui>
{
    typedef m512b bool_conv;
    typedef m512f float_conv;
    typedef m512i index_conv;
    typedef m512s pixel_conv;
    typedef ArrayUnion<m512ui> union_conv;
};

template<>
struct SimdConvTraits<m512us>
{
    typedef m512b bool_conv;
    typedef m512f float_conv;
    typedef m512i index_conv;
    typedef m512s pixel_conv;
    typedef ArrayUnion<m512us> union_conv;
};


template <typename V>
using FloatConv = typename SimdConvTraits<V>::float_conv;

template <typename V>
using IndexConv = typename SimdConvTraits<V>::index_conv;

template <typename V>
using BoolConv = typename SimdConvTraits<V>::bool_conv;

template <typename V>
using PixelConv = typename SimdConvTraits<V>::pixel_conv;

template <typename V>
using UnionConv = typename SimdConvTraits<V>::union_conv;

template <typename V>
UnionConv<V> MakeUnion(const V& v) { return UnionConv<V>(v); }

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_SimdConvTraits_H_

