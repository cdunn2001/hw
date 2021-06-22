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

#ifndef PACBIO_BAZIO_ENCODING_BAZ_TO_PULSE_H
#define PACBIO_BAZIO_ENCODING_BAZ_TO_PULSE_H

#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/FieldAccessors.h>
#include <bazio/encoding/EncodingParams.h>
#include <bazio/encoding/FieldTransforms.h>
#include <bazio/encoding/FieldSerializers.h>

#include <common/utility/Overload.h>

namespace PacBio {
namespace BazIO {

// Proof of concept class for converting between baz byte streams and pulses.
// While the class is named BazToPulse, it does currently handle both directions.
// It also deserializes directly into a Pulse structure, instead of a vector of
// vectors as PPA does it.  Both of those facts will likely change in the future,
// and the reason this class targets converting Baz files to pulse data is because
// it's chosen strategy relies on switch and variant visitation in order to be
// able to handle any valid baz encoding format.  It's about 6-7x slower than a
// serializer hard coded to handle one specific format, but this way if we tweak
// the exact data the basewriter is trying to serialize, we don't have to rebuild
// ppa and we don't have to worry about a lot of subtle binary incompatabilities.
// A single version of this parser will be able to handle any configuration as long
// as we don't actually change the set of fieldnames possible to serialize, or any
// of the serialization/transformation strategies supported.
class BazToPulse
{
public:
    // Accepts a encoding specification, complete with the
    // heirarchy of which fields are grouped together
    // (e.g. to handle cross byte fields)
    BazToPulse(std::vector<GroupParams> groups)
        : groups_(std::move(groups))
    {}

    // Accepts a flat list of fields to be serialized.  It will form the
    // groups itself, choosing the smallest groups that fit into integral
    // bytes
    BazToPulse(const std::vector<FieldParams>& info)
    {
        GroupParams currGroup;
        for (size_t i = 0; i < info.size(); ++i)
        {
            auto numBits = boost::apply_visitor([](const auto& v){ return v.numBits; },
                                                info[i].serialize);
            if ((currGroup.totalBits > 0 && currGroup.totalBits % 8 == 0)
                || currGroup.totalBits + numBits > 64)
            {
                groups_.push_back(std::move(currGroup));
                currGroup = GroupParams{};
            }
            currGroup.totalBits += numBits;
            assert(currGroup.totalBits <= 64);
            currGroup.members.push_back(info[i]);
            currGroup.numBits.push_back(numBits);
        }
        groups_.push_back(currGroup);
    }

    // Clears all state data.  Necessary if you walked through a bunch of
    // pulses calling `BytesRequired` to figure out how much memory to allocate
    void Reset()
    {
        *this = BazToPulse(std::move(groups_));
    }

    template <typename P>
    uint8_t* Serialize(const P& pulse, uint8_t* dest)
    {
        // Helper to handle the transformation visitation
        auto convertFloat = [](const TransformsParams& info, float val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const FixedPointParams& p) { return FixedPoint::Apply(val, storeSigned, p.scale); },
                [&](const auto&) -> uint64_t { throw PBException("Transformation does not support floating point values"); return 0; }
            );
            return boost::apply_visitor(o, info);
        };
        auto convertUInt = [](const TransformsParams& info, uint64_t val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const NoOpTransformParams&) { return NoOp::Apply(val, storeSigned); },
                [&](const CodecParams& p) { return LossySequelCodec::Apply(val, storeSigned, p.numBits); },
                [&](const FixedPointParams&) -> uint64_t { throw PBException("Transformation does not support integral values"); return 0;}
            );
            return boost::apply_visitor(o, info);
        };

        // Helper to handle the serialization visitation
        auto encode = [](const SerializeParams& info, uint64_t val, StoreSigned storeSigned, uint8_t*& ptr)
        {
            auto o = Utility::make_overload(
                [&](const TruncateParams& info) { return TruncateOverflow::ToBinary(val, ptr, storeSigned, info.numBits); },
                [&](const CompactOverflowParams& info) { return CompactOverflow::ToBinary(val, ptr, storeSigned, info.numBits); },
                [&](const SimpleOverflowParams& info) { return SimpleOverflow::ToBinary(val, ptr, storeSigned, info.numBits, info.overflowBytes); }
            );
            return boost::apply_visitor(o, info);
        };

        for (const auto& g : groups_)
        {
            size_t pos = 0;
            size_t mainBytes = (g.totalBits + 7) / 8;
            uint64_t mainVal = 0;
            auto oPtr = dest + mainBytes;
            for (size_t i = 0; i < g.members.size(); ++i)
            {
                const auto& info = g.members[i];
                auto numBits = g.numBits[i];
                uint64_t val = 0;
                switch (info.name)
                {
                case PacketFieldName::Base:
                    val = convertUInt(info.transform, base.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pw:
                    val = convertUInt(info.transform, pw.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Ipd:
                    val = convertUInt(info.transform, ipd.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkmax:
                    val = convertFloat(info.transform, pkmax.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkmean:
                    val = convertFloat(info.transform, pkmean.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkmid:
                    val = convertFloat(info.transform, pkmid.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkvar:
                    val = convertFloat(info.transform, pkvar.Get(pulse), info.storeSigned);
                    break;
                default:
                    throw PBException("FieldName Not Supported");
                }
                val = encode(info.serialize, val, info.storeSigned, oPtr);
                mainVal |= (val & ((1 << numBits) - 1)) << pos;
                pos += numBits;
            }
            std::memcpy(dest, &mainVal, mainBytes);
            dest = oPtr;
        }
        return dest;
    }

    template <typename P>
    uint8_t* Deserialize(P& pulse, uint8_t* source)
    {
        // Helpers to handle the transformation visitation
        auto convertUint = [&](const TransformsParams& info, uint64_t val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const NoOpTransformParams&) { return NoOp::Revert<uint64_t>(val, storeSigned); },
                [&](const CodecParams& p) { return LossySequelCodec::Revert<uint64_t>(val, storeSigned, p.numBits); },
                [&](const FixedPointParams&) -> uint64_t { throw PBException("Datatype does not support FixedPoint format"); }
            );
            return boost::apply_visitor(o, info);
        };
        auto convertFloat = [&](const TransformsParams& info, uint64_t val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const FixedPointParams& p) { return FixedPoint::Revert<float>(val, storeSigned, p.scale); },
                [&](const auto&) -> float { throw PBException("Datatype does not support FixedPoint format"); }
            );
            return boost::apply_visitor(o, info);
        };

        // Helper to handle the serialization visitation
        auto decode = [](const SerializeParams& info, uint64_t val, auto& ptr, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const TruncateParams& info) { return TruncateOverflow::FromBinary(val, ptr, storeSigned, info.numBits); },
                [&](const CompactOverflowParams& info) { return CompactOverflow::FromBinary(val, ptr, storeSigned, info.numBits); },
                [&](const SimpleOverflowParams& info) { return SimpleOverflow::FromBinary(val, ptr, storeSigned, info.numBits, info.overflowBytes); }
            );
            return boost::apply_visitor(o, info);
        };

        for (const auto& g : groups_)
        {
            size_t mainBytes = (g.totalBits + 7) / 8;
            uint64_t mainVal = 0;
            memcpy(&mainVal, source, mainBytes);
            source += mainBytes;
            for (size_t i = 0; i < g.members.size(); ++i)
            {
                const auto& info = g.members[i];
                auto numBits = g.numBits[i];
                uint64_t val = mainVal & ((1 << numBits)-1);
                mainVal = mainVal >> numBits;
                auto integral = decode(info.serialize, val, source, info.storeSigned);
                switch (info.name)
                {
                    case PacketFieldName::Base:
                        base.Set(pulse, convertUint(info.transform, integral, info.storeSigned));
                        break;
                    case PacketFieldName::Pw:
                        pw.Set(pulse, convertUint(info.transform, integral, info.storeSigned));
                        break;
                    case PacketFieldName::Ipd:
                        ipd.Set(pulse, convertUint(info.transform, integral, info.storeSigned));
                        break;
                    case PacketFieldName::Pkmax:
                        pkmax.Set(pulse, convertFloat(info.transform, integral, info.storeSigned));
                        break;
                    case PacketFieldName::Pkmean:
                        pkmean.Set(pulse, convertFloat(info.transform, integral, info.storeSigned));
                        break;
                    case PacketFieldName::Pkmid:
                        pkmid.Set(pulse, convertFloat(info.transform, integral, info.storeSigned));
                        break;
                    case PacketFieldName::Pkvar:
                        pkvar.Set(pulse, convertFloat(info.transform, integral, info.storeSigned));
                        break;
                    default:
                        throw PBException("FieldName Not Supported");
                }
            }
        }
        return source;
    }

    template <typename P>
    size_t BytesRequired(const P& pulse)
    {
        // Helper to handle the transformation visitation
        auto convertFloat = [](const TransformsParams& info, float val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const FixedPointParams& p) { return FixedPoint::Apply(val, storeSigned, p.scale); },
                [&](const auto&) -> uint64_t { throw PBException("Transformation does not support floating point values"); return 0; }
            );
            return boost::apply_visitor(o, info);
        };
        auto convertUInt = [](const TransformsParams& info, uint64_t val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const NoOpTransformParams&) { return NoOp::Apply(val, storeSigned); },
                [&](const CodecParams& p) { return LossySequelCodec::Apply(val, storeSigned, p.numBits); },
                [&](const FixedPointParams&) -> uint64_t { throw PBException("Transformation does not support integral values"); return 0;}
            );
            return boost::apply_visitor(o, info);
        };

        // Helper to handle the serialization visitation
        auto encode = [](const SerializeParams& info, uint64_t val, StoreSigned storeSigned)
        {
            auto o = Utility::make_overload(
                [&](const TruncateParams& info) { return TruncateOverflow::OverflowBytes(val, storeSigned, info.numBits); },
                [&](const CompactOverflowParams& info) { return CompactOverflow::OverflowBytes(val, storeSigned, info.numBits); },
                [&](const SimpleOverflowParams& info) { return SimpleOverflow::OverflowBytes(val, storeSigned, info.numBits, info.overflowBytes); }
            );
            return boost::apply_visitor(o, info);
        };

        size_t bytes = 0;
        for (const auto& g : groups_)
        {
            bytes += (g.totalBits + 7) / 8;
            for (const auto& info : g.members)
            {
                uint64_t val;
                switch (info.name)
                {
                case PacketFieldName::Base:
                    val = convertUInt(info.transform, base.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pw:
                    val = convertUInt(info.transform, pw.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Ipd:
                    val = convertUInt(info.transform, ipd.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkmax:
                    val = convertFloat(info.transform, pkmax.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkmean:
                    val = convertFloat(info.transform, pkmean.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkmid:
                    val = convertFloat(info.transform, pkmid.Get(pulse), info.storeSigned);
                    break;
                case PacketFieldName::Pkvar:
                    val = convertFloat(info.transform, pkvar.Get(pulse), info.storeSigned);
                    break;
                default:
                    throw PBException("FieldName Not Supported");
                }
                bytes += encode(info.serialize, val, info.storeSigned);
            }
        }
        return bytes;
    }
private:

    std::vector<GroupParams> groups_;
    PulseFieldAccessor<PacketFieldName::Base> base;
    PulseFieldAccessor<PacketFieldName::Pw> pw;
    PulseFieldAccessor<PacketFieldName::Ipd> ipd;
    PulseFieldAccessor<PacketFieldName::Pkmax> pkmax;
    PulseFieldAccessor<PacketFieldName::Pkmean> pkmean;
    PulseFieldAccessor<PacketFieldName::Pkmid> pkmid;
    PulseFieldAccessor<PacketFieldName::Pkvar> pkvar;
};

}}

#endif //PACBIO_BAZIO_ENCODING_BAZ_TO_PULSE_H
