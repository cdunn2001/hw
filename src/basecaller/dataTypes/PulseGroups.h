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

#ifndef PACBIO_MONGO_DATA_PULSE_GROUPS_H
#define PACBIO_MONGO_DATA_PULSE_GROUPS_H

#include <bazio/encoding/ObjectToBaz.h>

namespace PacBio {
namespace Mongo {
namespace Data {

using ProductionPulses =
    BazIO::ObjectToBaz<BazIO::Field<BazIO::PacketFieldName::Label,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::NoOp>,
                                    BazIO::Serialize<BazIO::TruncateOverflow, BazIO::NumBits_t<2>>>,
                       BazIO::Field<BazIO::PacketFieldName::PulseWidth,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::NoOp>,
                                    BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>,
                       BazIO::Field<BazIO::PacketFieldName::StartFrame,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::DeltaCompression>,
                                    BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>>;

using InternalPulses =
    BazIO::ObjectToBaz<BazIO::Field<BazIO::PacketFieldName::Label,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::NoOp>,
                                    BazIO::Serialize<BazIO::TruncateOverflow, BazIO::NumBits_t<2>>>,
                       BazIO::Field<BazIO::PacketFieldName::PulseWidth,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::NoOp>,
                                    BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>,
                       BazIO::Field<BazIO::PacketFieldName::StartFrame,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::DeltaCompression>,
                                    BazIO::Serialize<BazIO::CompactOverflow, BazIO::NumBits_t<7>>>,
                       BazIO::Field<BazIO::PacketFieldName::Pkmax,
                                    BazIO::StoreSigned_t<true>,
                                    BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                    BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<8>, BazIO::NumBytes_t<2>>>,
                       BazIO::Field<BazIO::PacketFieldName::Pkmid,
                                    BazIO::StoreSigned_t<true>,
                                    BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                    BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<8>, BazIO::NumBytes_t<2>>>,
                       BazIO::Field<BazIO::PacketFieldName::Pkmean,
                                    BazIO::StoreSigned_t<true>,
                                    BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                    BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<8>, BazIO::NumBytes_t<2>>>,
                       BazIO::Field<BazIO::PacketFieldName::Pkvar,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::FixedPoint, BazIO::FixedPointScale_t<10>>,
                                    BazIO::Serialize<BazIO::SimpleOverflow, BazIO::NumBits_t<7>, BazIO::NumBytes_t<2>>>,
                       BazIO::Field<BazIO::PacketFieldName::IsBase,
                                    BazIO::StoreSigned_t<false>,
                                    BazIO::Transform<BazIO::NoOp>,
                                    BazIO::Serialize<BazIO::TruncateOverflow, BazIO::NumBits_t<1>>>>;
}}} // PacBio::Mongo::Data

#endif //PACBIO_MONGO_DATA_PULSE_GROUPS_H
