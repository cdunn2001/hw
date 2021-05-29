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
//
/// \brief Helper classes that can be used to extract a given field from a pulse,
///        based off a specific PacketFieldName
//
//  Note: These classes at least partially exist because we need something stateful
//        to convert StartFrame to IPD.  It's crossed my mind that we could
//        potentially let the pulse accumulator handle that conversion, and just
//        store the IPD in the pulse rather than start frame.  Of course, we'd have
//        to update the stored IPDs once we start enabling any form of pulse exclusion,
//        and there might be other issues to deal with as well.  I'm not ready to
//        untangle all that just now, so leaving this breadcrumb here to make me, or
//        anyone else, think about it more in the future.

#ifndef PACBIO_BAZIO_ENCODING_FIELD_ACCESSORS_H
#define PACBIO_BAZIO_ENCODING_FIELD_ACCESSORS_H

#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/BazioEnableCuda.h>

namespace PacBio {
namespace BazIO {

template <PacketFieldName::RawEnum> struct PulseFieldAccessor;
template <>
struct PulseFieldAccessor<PacketFieldName::Base>
{
    using Type = uint8_t;
    template <typename P>
    BAZ_CUDA Type Get(const P& p) { return static_cast<Type>(p.Label()); }
    template <typename P>
    BAZ_CUDA void Set(P& p, Type val) { p.Label(static_cast<typename P::NucleotideLabel>(val)); }
};
template <>
struct PulseFieldAccessor<PacketFieldName::Ipd>
{
    using Type = uint32_t;
    template <typename P>
    BAZ_CUDA Type Get(const P& p)
    {
        auto ret = p.Start() - currFrame;
        currFrame = p.Start();
        return ret;
    }

    template <typename P>
    BAZ_CUDA void Set(P& p, Type val)
    {
        currFrame = val + currFrame;
        p.Start(currFrame);
    }

private:
    size_t currFrame = 0;
};
template <>
struct PulseFieldAccessor<PacketFieldName::Pw>
{
    using Type = uint32_t;
    template <typename P>
    BAZ_CUDA Type Get(const P& p) { return static_cast<Type>(p.Width()); }
    template <typename P>
    BAZ_CUDA void Set(P& p, const Type& val) { p.Width(val); }
};
template <>
struct PulseFieldAccessor<PacketFieldName::Pkmax>
{
    using Type = float;
    template <typename P>
    BAZ_CUDA Type Get(const P& p) { return static_cast<Type>(p.MaxSignal()); }
    template <typename P>
    BAZ_CUDA void Set(P& p, const Type& val) { p.MaxSignal(val); }
};
template <>
struct PulseFieldAccessor<PacketFieldName::Pkmid>
{
    using Type = float;
    template <typename P>
    BAZ_CUDA Type Get(const P& p) { return static_cast<Type>(p.MidSignal()); }
    template <typename P>
    BAZ_CUDA void Set(P& p, const Type& val) { p.MidSignal(val); }
};
template <>
struct PulseFieldAccessor<PacketFieldName::Pkmean>
{
    using Type = float;
    template <typename P>
    BAZ_CUDA Type Get(const P& p) { return static_cast<Type>(p.MeanSignal()); }
    template <typename P>
    BAZ_CUDA void Set(P& p, const Type& val) { p.MeanSignal(val); }
};
template <>
struct PulseFieldAccessor<PacketFieldName::Pkvar>
{
    using Type = float;
    template <typename P>
    BAZ_CUDA Type Get(const P& p) { return static_cast<Type>(p.SignalM2()); }
    template <typename P>
    BAZ_CUDA void Set(P& p, const Type& val) { p.SignalM2(val); }
};

}}

#endif //PACBIO_BAZIO_ENCODING_FIELDNAMES_H
