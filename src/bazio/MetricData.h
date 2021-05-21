// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_POSTPRIMARY_METRICCOLLECTIONS_H
#define PACBIO_POSTPRIMARY_METRICCOLLECTIONS_H

#include <utility>
#include <vector>

namespace PacBio {
namespace Primary {

template <class Storage>
struct AnalogMetricData
{
    Storage A;
    Storage C;
    Storage T;
    Storage G;

    // A little SFINAE magic to make sure this function only exists when the
    // class is templated on a type that supports this API
    template <typename S = Storage>
    decltype(std::declval<S>().size())
    size() const 
    {
        assert(A.size() == C.size());
        assert(A.size() == G.size());
        assert(A.size() == T.size());
        return A.size();
    }

    template <typename S = Storage>
    decltype(std::declval<S>().empty())
    empty() const
    {
        assert(A.size() == C.size());
        assert(A.size() == G.size());
        assert(A.size() == T.size());
        return A.empty();
    }

    template <typename S = Storage>
    decltype(std::declval<S>().capacity())
    capacity() const
    {
        assert(A.capacity() == C.capacity());
        assert(A.capacity() == G.capacity());
        assert(A.capacity() == T.capacity());
        return A.capacity();
    }

    template <typename S = Storage, std::enable_if_t<std::is_arithmetic<S>::value, int> = 0>
    size_t EstimatedBytesUsed() const
    {
        return 4*sizeof(S);
    }

    template <typename S = Storage>
    decltype(std::declval<S>().capacity())
    EstimatedBytesUsed() const
    {
        return 4*sizeof(S)*A.capacity();
    }
};

template <class Storage>
struct FilterMetricData
{
    
    Storage green;
    Storage red;

    // A little SFINAE magic to make sure this function only exists when the
    // class is templated on a type that suports this API
    template <typename S = Storage>
    decltype(std::declval<S>().size())
    size() const 
    {
        assert(green.size() == red.size());
        return green.size();
    }

    template <typename S = Storage>
    decltype(std::declval<S>().empty())
    empty() const
    {
        assert(green.size() == red.size());
        return green.empty();
    }

    template <typename S = Storage>
    decltype(std::declval<S>().capacity())
    capacity() const
    {
        assert(green.capacity() == red.capacity());
        return green.capacity();
    }

    template <typename S = Storage, std::enable_if_t<std::is_arithmetic<S>::value, int> = 0>
    size_t EstimatedBytesUsed() const
    {
        return 2*sizeof(S);
    }

    template <typename S = Storage>
    decltype(std::declval<S>().capacity())
    EstimatedBytesUsed() const
    {
        return 2*sizeof(S)*green.capacity();
    }

    explicit operator AnalogMetricData<Storage>() const
    {
        AnalogMetricData<Storage> ret;

        ret.A = red;
        ret.C = red;
        ret.G = green;
        ret.T = green;

        return ret;
    }
};

// Helper functions since things like SNR are defined as the ratio of two other
// metrics
inline AnalogMetricData<float> operator*(const AnalogMetricData<float>& left,
                                         const AnalogMetricData<float>& right)
{
    AnalogMetricData<float> ret;
    ret.A = left.A * right.A;
    ret.C = left.C * right.C;
    ret.G = left.G * right.G;
    ret.T = left.T * right.T;
    return ret;
}
inline AnalogMetricData<float> operator/(const AnalogMetricData<float>& left,
                                         const AnalogMetricData<float>& right)
{
    AnalogMetricData<float> ret;
    ret.A = left.A / right.A;
    ret.C = left.C / right.C;
    ret.G = left.G / right.G;
    ret.T = left.T / right.T;
    return ret;
}

}}

#endif /* PACBIO_POSTPRIMARY_METRICCOLLECTIONS_H */

