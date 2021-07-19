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
#include <functional>

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
    void CreateHelpers()
    {
        // Helper to handle the transformation visitation
        auto transformFloat = [](const TransformsParams& info)
        {
            using Ret = std::function<uint64_t(float, StoreSigned)>;
            return info.params.Visit(
                [&](const FixedPointParams& p) -> Ret { return [p](float f, StoreSigned storeSigned) { return FixedPoint::Apply(f, storeSigned, FixedPointScale{p.scale}); }; },
                [&](const auto&) -> Ret { throw PBException("Transformation does not support floating point values"); return 0; }
            );
        };
        auto transformUInt = [](const TransformsParams& info)
        {
            using Ret = std::function<uint64_t(uint64_t, StoreSigned)>;
            return info.params.Visit(
                [&](const NoOpTransformParams&) -> Ret { return [](uint64_t v, StoreSigned storeSigned) { return NoOp::Apply(v, storeSigned); }; },
                [&](const DeltaCompressionParams&) -> Ret { return [d = DeltaCompression{}](uint64_t v, StoreSigned storeSigned) mutable { return d.Apply(v, storeSigned); }; },
                [&](const CodecParams& p) -> Ret { return [p](uint64_t v, StoreSigned storeSigned) { return LossySequelCodec::Apply(v, storeSigned, NumBits{p.numBits}); }; },
                [&](const FixedPointParams&) -> Ret { throw PBException("Transformation does not support integral values"); return 0;}
            );
        };

        auto revertFloat = [](const TransformsParams& info)
        {
            using Ret = std::function<float(uint64_t, StoreSigned)>;
            return info.params.Visit(
                [&](const FixedPointParams& p) -> Ret { return [p](uint64_t v, StoreSigned storeSigned) { return FixedPoint::Revert<float>(v, storeSigned, FixedPointScale{p.scale}); }; },
                [&](const auto&) -> Ret { throw PBException("Transformation does not support floating point values"); return 0; }
            );
        };
        auto revertUInt = [](const TransformsParams& info)
        {
            using Ret = std::function<uint64_t(uint64_t, StoreSigned)>;
            return info.params.Visit(
                [&](const NoOpTransformParams&) -> Ret { return [](uint64_t v, StoreSigned storeSigned) { return NoOp::Revert<uint64_t>(v, storeSigned); }; },
                [&](const DeltaCompressionParams&) -> Ret { return [d = DeltaCompression{}](uint64_t v, StoreSigned storeSigned) mutable { return d.Revert<uint64_t>(v, storeSigned); }; },
                [&](const CodecParams& p) -> Ret { return [p](uint64_t v, StoreSigned storeSigned) { return LossySequelCodec::Revert<uint64_t>(v, StoreSigned{storeSigned}, NumBits{p.numBits}); }; },
                [&](const FixedPointParams&) -> Ret { throw PBException("Transformation does not support integral values"); return 0;}
            );
        };
        for (const auto& g : groups_)
        {
            for (const auto& f : g.members)
            {
                const auto info = f.transform;
                std::function<uint64_t(uint64_t, StoreSigned)> tmp;
                switch(f.name)
                {
                case PacketFieldName::Label:
                    base.transform_ = transformUInt(info[0]);
                    base.revert_ = revertUInt(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        base.transform_ = [f = base.transform_, tmp](uint64_t val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        base.revert_ = [f = base.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::IsBase:
                    isBase.transform_ = transformUInt(info[0]);
                    isBase.revert_ = revertUInt(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        isBase.transform_ = [f = isBase.transform_, tmp](uint64_t val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        isBase.revert_ = [f = isBase.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::StartFrame:
                    startFrame.transform_ = transformUInt(info[0]);
                    startFrame.revert_ = revertUInt(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        startFrame.transform_ = [f = startFrame.transform_, tmp](uint64_t val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        startFrame.revert_ = [f = startFrame.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::Pw:
                    pw.transform_ = transformUInt(info[0]);
                    pw.revert_ = revertUInt(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        pw.transform_ = [f = pw.transform_, tmp](uint64_t val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        pw.revert_ = [f = pw.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::Pkmax:
                    pkmax.transform_ = transformFloat(info[0]);
                    pkmax.revert_ = revertFloat(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        pkmax.transform_ = [f = pkmax.transform_, tmp](float val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        pkmax.revert_ = [f = pkmax.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::Pkmid:
                    pkmid.transform_ = transformFloat(info[0]);
                    pkmid.revert_ = revertFloat(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        pkmid.transform_ = [f = pkmid.transform_, tmp](float val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        pkmid.revert_ = [f = pkmid.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::Pkmean:
                    pkmean.transform_ = transformFloat(info[0]);
                    pkmean.revert_ = revertFloat(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        pkmean.transform_ = [f = pkmean.transform_, tmp](float val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        pkmean.revert_ = [f = pkmean.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                case PacketFieldName::Pkvar:
                    pkvar.transform_ = transformFloat(info[0]);
                    pkvar.revert_ = revertFloat(info[0]);
                    for (size_t i = 1; i < info.size(); ++i)
                    {
                        tmp = transformUInt(info[i]);
                        pkvar.transform_ = [f = pkvar.transform_, tmp](float val, StoreSigned storeSigned){ return tmp(f(val, storeSigned), storeSigned); };
                        tmp = revertUInt(info[i]);
                        pkvar.revert_ = [f = pkvar.revert_, tmp](uint64_t val, StoreSigned storeSigned){ return f(tmp(val, storeSigned), storeSigned); };
                    }
                    break;
                }
            }
        }
    }
public:
    // Accepts a encoding specification, complete with the
    // heirarchy of which fields are grouped together
    // (e.g. to handle cross byte fields)
    BazToPulse(std::vector<GroupParams> groups)
        : groups_(std::move(groups))
    {
        CreateHelpers();
    }

    // Accepts a flat list of fields to be serialized.  It will form the
    // groups itself, choosing the smallest groups that fit into integral
    // bytes
    BazToPulse(const std::vector<FieldParams>& info)
    {
        GroupParams currGroup;
        for (size_t i = 0; i < info.size(); ++i)
        {
            auto numBits = info[i].serialize.params.Visit([](const auto& v) { return v.numBits; });
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
        CreateHelpers();
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
        // Helper to handle the serialization visitation
        auto encode = [](const SerializeParams& info, uint64_t val, StoreSigned storeSigned, uint8_t*& ptr)
        {
            return info.params.Visit(
                [&](const TruncateParams& info) { return TruncateOverflow::ToBinary(val, ptr, storeSigned, NumBits{info.numBits}); },
                [&](const CompactOverflowParams& info) { return CompactOverflow::ToBinary(val, ptr, storeSigned, NumBits{info.numBits}); },
                [&](const SimpleOverflowParams& info) { return SimpleOverflow::ToBinary(val, ptr, storeSigned, NumBits{info.numBits}, NumBytes{info.overflowBytes}); }
            );
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
                assert(!info.transform.empty());
                switch (info.name)
                {
                case PacketFieldName::Label:
                    val = base.transform_(base.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::IsBase:
                    val = isBase.transform_(isBase.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pw:
                    val = pw.transform_(pw.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::StartFrame:
                    val = startFrame.transform_(startFrame.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkmax:
                    val = pkmax.transform_(pkmax.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkmean:
                    val = pkmean.transform_(pkmean.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkmid:
                    val = pkmid.transform_(pkmid.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkvar:
                    val = pkvar.transform_(pkvar.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                default:
                    throw PBException("FieldName Not Supported");
                }
                val = encode(info.serialize, val, StoreSigned{info.storeSigned}, oPtr);
                mainVal |= (val & ((1 << numBits) - 1)) << pos;
                pos += numBits;
            }
            std::memcpy(dest, &mainVal, mainBytes);
            dest = oPtr;
        }
        return dest;
    }

    template <typename P>
    const uint8_t* Deserialize(P& pulse, const uint8_t* source)
    {
        // Helper to handle the serialization visitation
        auto decode = [](const SerializeParams& info, uint64_t val, auto const *& ptr, StoreSigned storeSigned)
        {
            return info.params.Visit(
                [&](const TruncateParams& info) { return TruncateOverflow::FromBinary(val, ptr, storeSigned, NumBits{info.numBits}); },
                [&](const CompactOverflowParams& info) { return CompactOverflow::FromBinary(val, ptr, storeSigned, NumBits{info.numBits}); },
                [&](const SimpleOverflowParams& info) { return SimpleOverflow::FromBinary(val, ptr, storeSigned, NumBits{info.numBits}, NumBytes{info.overflowBytes}); }
            );
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
                auto integral = decode(info.serialize, val, source, StoreSigned{info.storeSigned});
                assert(!info.transform.empty());
                switch (info.name)
                {
                    case PacketFieldName::Label:
                        base.access_.Set(pulse, base.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::IsBase:
                        isBase.access_.Set(pulse, isBase.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::Pw:
                        pw.access_.Set(pulse, pw.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::StartFrame:
                        startFrame.access_.Set(pulse, startFrame.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::Pkmax:
                        pkmax.access_.Set(pulse, pkmax.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::Pkmean:
                        pkmean.access_.Set(pulse, pkmean.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::Pkmid:
                        pkmid.access_.Set(pulse, pkmid.revert_(integral, StoreSigned{info.storeSigned}));
                        break;
                    case PacketFieldName::Pkvar:
                        pkvar.access_.Set(pulse, pkvar.revert_(integral, StoreSigned{info.storeSigned}));
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
        // Helper to handle the serialization visitation
        auto encode = [](const SerializeParams& info, uint64_t val, StoreSigned storeSigned)
        {
            return info.params.Visit(
                [&](const TruncateParams& info) { return TruncateOverflow::OverflowBytes(val, storeSigned, NumBits{info.numBits}); },
                [&](const CompactOverflowParams& info) { return CompactOverflow::OverflowBytes(val, storeSigned, NumBits{info.numBits}); },
                [&](const SimpleOverflowParams& info) { return SimpleOverflow::OverflowBytes(val, storeSigned, NumBits{info.numBits}, NumBytes{info.overflowBytes}); }
            );
        };

        size_t bytes = 0;
        for (const auto& g : groups_)
        {
            bytes += (g.totalBits + 7) / 8;
            for (const auto& info : g.members)
            {
                assert(!info.transform.empty());
                uint64_t val;
                switch (info.name)
                {
                case PacketFieldName::Label:
                    val = base.transform_(base.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::IsBase:
                    val = isBase.transform_(isBase.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pw:
                    val = pw.transform_(pw.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::StartFrame:
                    val = startFrame.transform_(startFrame.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkmax:
                    val = pkmax.transform_(pkmax.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkmean:
                    val = pkmean.transform_(pkmean.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkmid:
                    val = pkmid.transform_(pkmid.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                case PacketFieldName::Pkvar:
                    val = pkvar.transform_(pkvar.access_.Get(pulse), StoreSigned{info.storeSigned});
                    break;
                default:
                    throw PBException("FieldName Not Supported");
                }
                bytes += encode(info.serialize, val, StoreSigned{info.storeSigned});
            }
        }
        return bytes;
    }
private:
    template <PacketFieldName::RawEnum name>
    struct FieldHelpers
    {
        PulseFieldAccessor<name> access_;
        using T = typename PulseFieldAccessor<name>::Type;
        std::function<uint64_t(T, StoreSigned)> transform_;
        std::function<T(uint64_t, StoreSigned)> revert_;
    };

    std::vector<GroupParams> groups_;
    FieldHelpers<PacketFieldName::Label> base;
    FieldHelpers<PacketFieldName::IsBase> isBase;
    FieldHelpers<PacketFieldName::Pw> pw;
    FieldHelpers<PacketFieldName::StartFrame> startFrame;
    FieldHelpers<PacketFieldName::Pkmax> pkmax;
    FieldHelpers<PacketFieldName::Pkmean> pkmean;
    FieldHelpers<PacketFieldName::Pkmid> pkmid;
    FieldHelpers<PacketFieldName::Pkvar> pkvar;
};

}}

#endif //PACBIO_BAZIO_ENCODING_BAZ_TO_PULSE_H
