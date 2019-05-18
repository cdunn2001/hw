//
//  Copyright (c) 2011-2015, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//

/// \brief Implementations of some the fancy SIMD functions we implemented, for the
///        scalar case.

#pragma once

// TODO: Should generalize this to support scalar integer operations also.

namespace PacBio {

inline float add(const float& a, const float& b, const bool& mask)
{
    return (mask ? (a + b) : a);
}

inline float fmadd(const float& mul1, const float& mul2, const float& add1, const bool& mask)
{
    return add1 + (mask ? mul1 * mul2 : 0);
}

/// Template that supports float and int.
template <typename T>
inline T inc(const T& a, bool mask)
{
    return (mask ? (a + 1) : a);
}

inline float Blend(const bool& mask, const float& success, const float& failure)
{
    return (mask ? success : failure);
}

// http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
inline float
fasterlog2(float x)
{
    union
    {
        float f;
        uint32_t i;
    } vx = {x};
    float y = static_cast<float>(vx.i);
    y *= 1.0 / (1 << 23);
    return y - 126.94269504f;
}

inline float
fasterlog(float x)
{
    return 0.69314718f * fasterlog2(x);
}

inline float
fasterpow2(float p)
{
    union
    {
        uint32_t i;
        float f;
    } v;
    v.i = (1 << 23) * (p + 126.94269504f);
    return v.f;
}

inline float
expSuperFast(float p)
{
    return fasterpow2(1.442695040f * p);
}

inline int floorCastInt(float x)
{
    return static_cast<int>(std::floor(x));
}

}  // PacBio
