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
/// \File PulseGroups.h
/// \details Defines the following pulse group params representations:
///
///          * ProductionPulses:    Production pulse group params representation
///          * InternalPulses:      Internal pulse group params representation
///

#ifndef PACBIO_BAZIO_ENCODING_PULSE_GROUPS_H
#define PACBIO_BAZIO_ENCODING_PULSE_GROUPS_H

#include <bazio/encoding/PulseToBaz.h>

namespace PacBio {
namespace BazIO {

using ProductionPulses =
    PulseToBaz<Field<PacketFieldName::Base,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<TruncateOverflow, NumBits_t<2>>
                     >,
               Field<PacketFieldName::Pw,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >,
               Field<PacketFieldName::StartFrame,
                     StoreSigned_t<false>,
                     Transform<DeltaCompression>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >
               >;

using InternalPulses =
    PulseToBaz<Field<PacketFieldName::Base,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<TruncateOverflow, NumBits_t<2>>
                     >,
               Field<PacketFieldName::Pw,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >,
               Field<PacketFieldName::StartFrame,
                     StoreSigned_t<false>,
                     Transform<DeltaCompression>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >,
               Field<PacketFieldName::Pkmax,
                     StoreSigned_t<true>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<8>, NumBytes_t<2>>
                     >,
               Field<PacketFieldName::Pkmid,
                     StoreSigned_t<true>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<8>, NumBytes_t<2>>
                     >,
               Field<PacketFieldName::Pkmean,
                     StoreSigned_t<true>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<8>, NumBytes_t<2>>
                     >,
               Field<PacketFieldName::Pkvar,
                     StoreSigned_t<false>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<2>>
                     >,
               Field<PacketFieldName::IsBase,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<TruncateOverflow, NumBits_t<1>>
                     >
               >;

}}

#endif //PACBIO_BAZIO_ENCODING_PULSE_GROUPS_H
