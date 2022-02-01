#ifndef Sequel_Primary_Util_H_
#define Sequel_Primary_Util_H_

// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \file   Util.h
/// \brief  Utility functions for general use in Primary Analysis code.

#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace PacBio {
namespace Primary {

/// The number of independent elements in the (symmetric) covariance matrix of size nxn. 
constexpr size_t NumCvr(size_t n) { return n*(n + 1) / 2; }

/// Re-map and fill from array<float> to array<VF>
template <typename VF, size_t N>
std::array<VF, N> RemapAndFill(const std::array<float, N>& a, const std::array<int, N>& map)
{
    std::array<VF, N> r;
    for (unsigned int i = 0; i < N; ++i) r[i] = a[map[i]];
    return r;
}

/// The L1 norm of an amplitude vector, i.e. after clipping negative components to zero.
template <typename It>
static typename std::iterator_traits<It>::value_type L1Norm(It first, It last)
{
    typedef typename std::iterator_traits<It>::value_type FType;
    const FType zero { 0.0f };
    FType sv { zero };

    while (first != last)
    {
        using std::max; // accommodate float
        FType xi = max(*first, zero);
        sv = sv + xi;

        ++first;
    }

    return sv;
}

/// The L2 (Euclidean) norm of an amplitude vector, after clipping negative components to zero.
template <typename It>
static typename std::iterator_traits<It>::value_type L2Norm(It first, It last)
{
    typedef typename std::iterator_traits<It>::value_type FType;
    const FType zero { 0.0f };
    FType ss { zero };

    //return std::inner_product(first, last, first, ss);
    while (first != last)
    {
        using std::max; // accommodate float
        FType xi = max(*first, zero);
        ss = ss + xi * xi;

        ++first;
    }
    using std::sqrt; // accommodate float
    return sqrt(ss);
}

/// Normalize amplitude vector to an L1Norm of the given amplitude: sum(v_i) == A [=1].
template <typename VF, size_t N>
std::array<VF, N> L1Normalized(const std::array<VF, N>& s, VF a = VF(1.0f))
{
    std::array<VF, N> r(s);
    a = a / L1Norm(s.begin(), s.end());

    for (size_t i = 0; i < N; ++i)
        r[i] = a * r[i];

    return r;
}

/// A "specialization" of L1Normalized<VF, N> for N=1.
/// \returns std::array<VF, 1ul> {a}.
template <typename VF> inline
std::array<VF, 1ul> L1Normalized(const std::array<VF, 1ul>& s, VF a = VF(1.0f))
{
    (void) s;   // Avoid compiler warning.
    return {a};
}

/// Normalize amplitude vector to an L2Norm of the given amplitude: sqrt(v-dot-v) == A [=1].
template <typename VF, size_t N>
std::array<VF, N> L2Normalized(const std::array<VF, N>& s, VF a = VF(1.0f))
{
    std::array<VF, N> r(s);
    a = a / L2Norm(s.begin(), s.end());
    for (unsigned int i = 0; i < N; ++i) r[i] = a * r[i];
    return r;
}

/// Convert the values in the range [first, last) to a string delimited by ", ".
/// \tparam An input iterator type.
/// \param first refers to the first element of the range.
/// \param last refers to one past the end of the range.
/// \returns The range of values converted to string.
/// \note Requires an overload of stream insert operator (<<) for
/// std::iterator_traits<InIter>::reference.
template <typename InIter>
std::string ToString(InIter first, const InIter& last, const std::string delim = ", ")
{
    std::ostringstream oss;
    if (first != last) oss << *first++;
    while (first != last) oss << delim << *first++;
    return oss.str();
}

}} // ::PacBio::Primary

# endif // Sequel_Primary_Util_H_
