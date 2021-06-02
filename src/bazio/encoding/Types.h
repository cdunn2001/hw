// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_BAZIO_ENCODING_TYPES_H
#define PACBIO_BAZIO_ENCODING_TYPES_H

#include <cstdint>
#include <common/utility/StrongTypedef.h>

namespace PacBio {
namespace BazIO {

// Set up some strong typedefs, both to prevent
// accidental argument swaps, but also to make
// call-sites easier to read
PB_STRONG_TYPEDEF(bool, StoreSigned);
PB_STRONG_TYPEDEF(uint8_t, NumBits);
PB_STRONG_TYPEDEF(uint8_t, NumBytes);
PB_STRONG_TYPEDEF(uint32_t, FixedPointScale);

// Set up some machinery for a compile time version
// of the above strong typedefs
template <typename U, typename U::T v>
struct TemplateVal
{
    static constexpr U val = U(v);
};
template <typename U, typename U::T v>
constexpr U TemplateVal<U, v>::val;

template <bool b> using StoreSigned_t  = TemplateVal<StoreSigned, b>;
template <uint8_t v> using NumBits_t  = TemplateVal<NumBits, v>;
template <uint8_t v> using NumBytes_t = TemplateVal<NumBytes, v>;
template <uint32_t v> using FixedPointScale_t = TemplateVal<FixedPointScale, v>;

}}

#endif //PACBIO_BAZIO_ENCODING_TYPES_H
