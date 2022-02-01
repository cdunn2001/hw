#ifndef mongo_common_simd_m512f_SSE_H_
#define mongo_common_simd_m512f_SSE_H_

// Copyright (c) 2015,2020 Pacific Biosciences of California, Inc.
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
/// \file   m512f_SSE.h
/// \brief  SIMD 512-bit float vector (16 packed 32-bit float values) for SSE.

#if !defined(__SSE2__)
#error This type requires SSE2 intrinsics.
#endif

#include <cassert>
#include <immintrin.h>
#include <limits>
#include <ostream>

#include "m512b_SSE.h"
#include "m512i_SSE.h"
#include "m512ui_SSE.h"
#include "xcompile.h"

namespace PacBio {
namespace Simd {

// SIMD 512-bit float vector (16 packed 32-bit float values).
CLASS_ALIGNAS(16) m512f
{
public:     // Types
    typedef m512f type;

public:     // static constants
    // Number of floats in one m512f.
    static constexpr size_t size()
    {
        return sizeof(m512f) / sizeof(float);
    }

private:    // Implementation
    using ImplType = __m128;
    static const size_t implOffsetElems = sizeof(ImplType) / sizeof(float);

private:
    union
    {
        ImplType simd[4];
        float raw[16];
    } data;

public:     // Structors
    // Purposefully do not initialize v.
    m512f() = default;

    // Replicate scalar x across v.
    m512f(float x)
        : data{{_mm_set1_ps(x)
              , _mm_set1_ps(x)
              , _mm_set1_ps(x)
              , _mm_set1_ps(x)}}
    {}

    // Convenient for initializing from an simple integer, e.g., m512f(0).
    // Eigen 3.2.5 uses such expressions, which are ambiguous without this constructor.
    m512f(int32_t x) : m512f(static_cast<float>(x))
    {}
    m512f(uint32_t x) : m512f(static_cast<float>(x))
    {}

    // Load x from pointer px. px must be aligned to 16 bytes.
    m512f(const float *px)
        : data{{_mm_load_ps(px + 0*implOffsetElems)
              , _mm_load_ps(px + 1*implOffsetElems)
              , _mm_load_ps(px + 2*implOffsetElems)
              , _mm_load_ps(px + 3*implOffsetElems)}}
    {}

    // Copy constructor
    m512f(const m512f& v) = default;

    // Construct from native vector type
    m512f(ImplType v1, ImplType v2, ImplType v3, ImplType v4)
        : data{{v1, v2, v3, v4}}
    {}

    // Construct from 16 floats
    m512f(float f0, float f1, float f2, float f3, float f4, float f5, float f6,
          float f7, float f8, float f9, float f10, float f11, float f12,
          float f13, float f14, float f15)
        : data{{_mm_setr_ps(f0,  f1,  f2,  f3)
              , _mm_setr_ps(f4,  f5,  f6,  f7)
              , _mm_setr_ps(f8,  f9,  f10, f11)
              , _mm_setr_ps(f12, f13, f14, f15)}}
    {}

    m512f(const m512i& v)
        : data{{_mm_cvtepi32_ps(v.data1()),
                _mm_cvtepi32_ps(v.data2()),
                _mm_cvtepi32_ps(v.data3()),
                _mm_cvtepi32_ps(v.data4())}}
    {}

    explicit operator m512i() const
    {
        return m512i(_mm_cvtps_epi32(_mm_round_ps(data.simd[0],_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
                     _mm_cvtps_epi32(_mm_round_ps(data.simd[1],_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
                     _mm_cvtps_epi32(_mm_round_ps(data.simd[2],_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)),
                     _mm_cvtps_epi32(_mm_round_ps(data.simd[3],_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)));
    }

    // dancing around a lack of unsigned intrinsics in SSE. We
    // have to emulate this one
    m512f(const m512ui& v)
    {
        for (size_t i = 0; i < size(); ++i)
        {
            data.raw[i] = static_cast<float>(v[i]);
        }
    }

    // dancing around a lack of unsigned intrinsics in SSE. We
    // have to emulate this one
    explicit operator m512ui() const
    {
        alignas(m512ui) uint32_t retData[size()];
        for (size_t i = 0; i < size(); ++i)
        {
            retData[i] = static_cast<uint32_t>(data.raw[i]);
        }
        return m512ui(retData);
    }

public:     // Assignment
    m512f& operator=(const m512f& x) = default;

    // Assignment from scalar value
    m512f& operator=(float fv)
    {
        data.simd[0] = _mm_set1_ps(fv);
        data.simd[1] = _mm_set1_ps(fv);
        data.simd[2] = _mm_set1_ps(fv);
        data.simd[3] = _mm_set1_ps(fv);

        return *this;
    }

    // Compound assignment operators
    m512f& operator+=(const m512f& x)
    {
        data.simd[0] = _mm_add_ps(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_add_ps(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_add_ps(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_add_ps(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512f& operator-=(const m512f& x)
    {
        data.simd[0] = _mm_sub_ps(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_sub_ps(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_sub_ps(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_sub_ps(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512f& operator*=(const m512f& x)
    {
        data.simd[0] = _mm_mul_ps(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_mul_ps(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_mul_ps(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_mul_ps(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    m512f& operator/=(const m512f& x)
    {
        data.simd[0] = _mm_div_ps(this->data.simd[0], x.data.simd[0]);
        data.simd[1] = _mm_div_ps(this->data.simd[1], x.data.simd[1]);
        data.simd[2] = _mm_div_ps(this->data.simd[2], x.data.simd[2]);
        data.simd[3] = _mm_div_ps(this->data.simd[3], x.data.simd[3]);
        return *this;
    }

    // Unary minus operator.
    m512f operator-() const
    {
        const auto zero = _mm_setzero_ps();
        return m512f(_mm_sub_ps(zero, data.simd[0]),
                     _mm_sub_ps(zero, data.simd[1]),
                     _mm_sub_ps(zero, data.simd[2]),
                     _mm_sub_ps(zero, data.simd[3]));
    }

    // Return a scalar value
    float operator[](unsigned int i) const
    {
        assert(static_cast<size_t>(i) < this->size());
        return data.raw[i];
    }

    const ImplType& data1() const
    { return data.simd[0]; }

    const ImplType& data2() const
    { return data.simd[1]; }

    const ImplType& data3() const
    { return data.simd[2]; }

    const ImplType& data4() const
    { return data.simd[3]; }

public:     // Non-member (friend) functions

    friend std::ostream& operator << (std::ostream& stream, const m512f& vec)
    {
        stream << vec[0] << "\t" << vec[1] << "\t" << vec[2] << "\t" << vec[3]
               << "\t" << vec[4] << "\t" << vec[5] << "\t" << vec[6] << "\t"
               << vec[7] << "\t" << vec[8] << "\t" << vec[9] << "\t" << vec[10]
               << "\t" << vec[11] << "\t" << vec[12] << "\t" << vec[13] << "\t"
               << vec[14] << "\t" << vec[15];

        return stream;
    }

    friend m512b operator == (const m512f &l, const m512f &r)
    {
        return m512b(_mm_cmpeq_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_cmpeq_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_cmpeq_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_cmpeq_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator != (const m512f &l, const m512f &r)
    {
        return m512b(_mm_cmpneq_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_cmpneq_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_cmpneq_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_cmpneq_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator > (const m512f &l, const m512f &r)
    {
        return m512b(_mm_cmpgt_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_cmpgt_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_cmpgt_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_cmpgt_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator >= (const m512f &l, const m512f &r)
    {
        return m512b(_mm_cmpge_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_cmpge_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_cmpge_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_cmpge_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b operator < (const m512f &l, const m512f &r)
    {
        return m512b(_mm_cmplt_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_cmplt_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_cmplt_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_cmplt_ps(l.data.simd[3], r.data.simd[3]));
    }
    friend m512b operator <= (const m512f &l, const m512f &r)
    {
        return m512b(_mm_cmple_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_cmple_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_cmple_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_cmple_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512f operator + (const m512f &l, const m512f &r)
    {
        return m512f(_mm_add_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_add_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_add_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_add_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512f operator - (const m512f &l, const m512f &r)
    {
        return m512f(_mm_sub_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_sub_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_sub_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_sub_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512f operator * (const m512f &l, const m512f &r)
    {
        return m512f(_mm_mul_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_mul_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_mul_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_mul_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512f operator / (const m512f &l, const m512f &r)
    {
        return m512f(_mm_div_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_div_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_div_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_div_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512b isnan(const m512f& x)
    {
        const auto ff = m512f(-1.0f);
        return m512b(_mm_andnot_ps(_mm_cmpord_ps(x.data.simd[0], x.data.simd[0]), ff.data.simd[0]),
                     _mm_andnot_ps(_mm_cmpord_ps(x.data.simd[1], x.data.simd[1]), ff.data.simd[1]),
                     _mm_andnot_ps(_mm_cmpord_ps(x.data.simd[2], x.data.simd[2]), ff.data.simd[2]),
                     _mm_andnot_ps(_mm_cmpord_ps(x.data.simd[3], x.data.simd[3]), ff.data.simd[3]));
    }

    friend m512b isnotnan(const m512f& x)
    {
        return m512b(_mm_cmpord_ps(x.data.simd[0], x.data.simd[0]),
                     _mm_cmpord_ps(x.data.simd[1], x.data.simd[1]),
                     _mm_cmpord_ps(x.data.simd[2], x.data.simd[2]),
                     _mm_cmpord_ps(x.data.simd[3], x.data.simd[3]));
    }

    friend m512b isinf(const m512f& x)
    {
        const auto ff = m512f(-1.0f);
        return m512b(_mm_andnot_ps(_mm_isfinite_ps(x.data.simd[0]), ff.data.simd[0]),
                     _mm_andnot_ps(_mm_isfinite_ps(x.data.simd[1]), ff.data.simd[1]),
                     _mm_andnot_ps(_mm_isfinite_ps(x.data.simd[2]), ff.data.simd[2]),
                     _mm_andnot_ps(_mm_isfinite_ps(x.data.simd[3]), ff.data.simd[3]));
    }

    friend m512b isfinite(const m512f& x)
    {
        return m512b(_mm_isfinite_ps(x.data.simd[0]),
                     _mm_isfinite_ps(x.data.simd[1]),
                     _mm_isfinite_ps(x.data.simd[2]),
                     _mm_isfinite_ps(x.data.simd[3]));
    }

    friend m512b isnormal(const m512f& x)
    {
        return m512b(_mm_isnormal_ps(x.data.simd[0]),
                     _mm_isnormal_ps(x.data.simd[1]),
                     _mm_isnormal_ps(x.data.simd[2]),
                     _mm_isnormal_ps(x.data.simd[3]));
    }

    friend m512f min(const m512f& l, const m512f&r)
    {
        return m512f(_mm_min_ps(l.data.simd[0], r.data.simd[0]),
                     _mm_min_ps(l.data.simd[1], r.data.simd[1]),
                     _mm_min_ps(l.data.simd[2], r.data.simd[2]),
                     _mm_min_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend m512f max(const m512f& l, const m512f&r)
    {
        return m512f(_mm_max_ps(l.data.simd[0], r.data.simd[0]),
                      _mm_max_ps(l.data.simd[1], r.data.simd[1]),
                      _mm_max_ps(l.data.simd[2], r.data.simd[2]),
                      _mm_max_ps(l.data.simd[3], r.data.simd[3]));
    }

    friend float reduceMax(const m512f& a)
    {
        float ret = -std::numeric_limits<float>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::max(ret, a[i]);
        }
        return ret;
    }

    friend float reduceMin(const m512f& a)
    {
        float ret = std::numeric_limits<float>::max();
        for (size_t i = 0; i < size(); ++i)
        {
            ret = std::min(ret, a[i]);
        }
        return ret;
    }

    friend m512f sqrt(const m512f& x)
    {
        const auto mask = x == m512f(0);
        auto a = Blend(mask, m512f(1), x);
        a = m512f(_mm_sqrt_ps(a.data.simd[0]),
                  _mm_sqrt_ps(a.data.simd[1]),
                  _mm_sqrt_ps(a.data.simd[2]),
                  _mm_sqrt_ps(a.data.simd[3]));
        return Blend(mask, m512f(0), a);
    }

    friend m512f square(const m512f& x)
    { return x * x; }

    friend m512f cos(const m512f& x)
    {
        return m512f(_mm_cos_ps(x.data.simd[0]),
                     _mm_cos_ps(x.data.simd[1]),
                     _mm_cos_ps(x.data.simd[2]),
                     _mm_cos_ps(x.data.simd[3]));
    }

    friend m512f sin(const m512f& x)
    {
        return m512f(_mm_sin_ps(x.data.simd[0]),
                     _mm_sin_ps(x.data.simd[1]),
                     _mm_sin_ps(x.data.simd[2]),
                     _mm_sin_ps(x.data.simd[3]));
    }

    friend m512f atan2(const m512f &y, const m512f &x)
    {
        return m512f(_mm_atan2_ps(y.data.simd[0], x.data.simd[0]),
                     _mm_atan2_ps(y.data.simd[1], x.data.simd[1]),
                     _mm_atan2_ps(y.data.simd[2], x.data.simd[2]),
                     _mm_atan2_ps(y.data.simd[3], x.data.simd[3]));
    }

    friend m512f log(const m512f &x)
    {
        return m512f(_mm_log_ps(x.data.simd[0]),
                     _mm_log_ps(x.data.simd[1]),
                     _mm_log_ps(x.data.simd[2]),
                     _mm_log_ps(x.data.simd[3]));
    }

    friend m512f log2(const m512f& x)
    {
        return m512f(_mm_log2_ps(x.data.simd[0]),
                     _mm_log2_ps(x.data.simd[1]),
                     _mm_log2_ps(x.data.simd[2]),
                     _mm_log2_ps(x.data.simd[3]));
    }

    friend m512f exp(const m512f &x)
    {
        return m512f(_mm_exp_ps(x.data.simd[0]),
                     _mm_exp_ps(x.data.simd[1]),
                     _mm_exp_ps(x.data.simd[2]),
                     _mm_exp_ps(x.data.simd[3]));
    }

    friend m512f exp2(const m512f& x)
    {
        return m512f(_mm_exp2_ps(x.data.simd[0]),
                     _mm_exp2_ps(x.data.simd[1]),
                     _mm_exp2_ps(x.data.simd[2]),
                     _mm_exp2_ps(x.data.simd[3]));
    }

    friend m512f erf(const m512f &x)
    {
        return m512f(_mm_erf_ps(x.data.simd[0]),
                     _mm_erf_ps(x.data.simd[1]),
                     _mm_erf_ps(x.data.simd[2]),
                     _mm_erf_ps(x.data.simd[3]));
    }

    /// Complementary error function for SSE.
    friend m512f erfc(const m512f& x)
    {
        return m512f(_mm_erfc_ps(x.data.simd[0]),
                     _mm_erfc_ps(x.data.simd[1]),
                     _mm_erfc_ps(x.data.simd[2]),
                     _mm_erfc_ps(x.data.simd[3]));
    }

    friend m512f erfDiff(const m512f &x, const m512f &y)
    {
        return m512f(_mm_sub_ps(_mm_erf_ps(x.data.simd[0]), _mm_erf_ps(y.data.simd[0])),
                     _mm_sub_ps(_mm_erf_ps(x.data.simd[1]), _mm_erf_ps(y.data.simd[1])),
                     _mm_sub_ps(_mm_erf_ps(x.data.simd[2]), _mm_erf_ps(y.data.simd[2])),
                     _mm_sub_ps(_mm_erf_ps(x.data.simd[3]), _mm_erf_ps(y.data.simd[3])));
    }

    friend m512f Blend(const m512b& mask, const m512f& success, const m512f& failure)
    {
        return m512f(_mm_blendv_ps(failure.data.simd[0], success.data.simd[0], mask.data1()),
                     _mm_blendv_ps(failure.data.simd[1], success.data.simd[1], mask.data2()),
                     _mm_blendv_ps(failure.data.simd[2], success.data.simd[2], mask.data3()),
                     _mm_blendv_ps(failure.data.simd[3], success.data.simd[3], mask.data4()));
    }

    friend m512f inc(const m512f& a, const m512b& mask)
    {
        return Blend(mask, a + m512f(1) , a);
    }
};

inline m512i floorCastInt(const m512f& f)
{
    return m512i(_mm_cvtps_epi32(_mm_floor_ps(f.data1()))
                ,_mm_cvtps_epi32(_mm_floor_ps(f.data2()))
                ,_mm_cvtps_epi32(_mm_floor_ps(f.data3()))
                ,_mm_cvtps_epi32(_mm_floor_ps(f.data4())));
}

inline m512ui floorCastUInt(const m512f& f)
{
    return m512ui(_mm_cvtps_epi32(_mm_floor_ps(f.data1()))
                 ,_mm_cvtps_epi32(_mm_floor_ps(f.data2()))
                 ,_mm_cvtps_epi32(_mm_floor_ps(f.data3()))
                 ,_mm_cvtps_epi32(_mm_floor_ps(f.data4())));
}

inline m512i roundCastInt(const m512f& f)
{
    constexpr int mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    return m512i(_mm_cvtps_epi32(_mm_round_ps(f.data1(), mode)),
                 _mm_cvtps_epi32(_mm_round_ps(f.data2(), mode)),
                 _mm_cvtps_epi32(_mm_round_ps(f.data3(), mode)),
                 _mm_cvtps_epi32(_mm_round_ps(f.data4(), mode)));
}

}}      // namespace PacBio::Simd

#endif  // mongo_common_simd_m512f_SSE_H_
