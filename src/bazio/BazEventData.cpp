// Copyright (c) 2018,2021, Pacific Biosciences of California, Inc.
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

#include "BazEventData.h"

#include <bazio/encoding/FieldTransforms.h>
#include <common/utility/Overload.h>

namespace PacBio {
namespace BazIO {
namespace {

    // Deserializes integral fields (where things like the FixedPoint transformation
    // makes no sense)
    template <typename TOut>
    std::vector<TOut> ConvertIntegral(const std::vector<uint32_t>& raw,
                                      const BazIO::TransformsParams& info,
                                      BazIO::PacketFieldName name,
                                      BazIO::StoreSigned storeSigned)
    {
        static_assert(std::is_integral<TOut>::value, "Output type should be integral");

        auto noop = [&](const BazIO::NoOpTransformParams&) {
            std::vector<TOut> ret;
            ret.reserve(raw.size());
            for (const auto& val : raw)
                ret.push_back(BazIO::NoOp::Revert<TOut>(val, storeSigned));
            return ret;
        };
        auto codec = [&](const BazIO::CodecParams& params) {
            std::vector<TOut> ret;
            ret.reserve(raw.size());
            for (const auto& val : raw)
                ret.push_back(BazIO::LossySequelCodec::Revert<TOut>(val, storeSigned, static_cast<BazIO::NumBits>(params.numBits)));
            return ret;
        };
        auto delta = [&](const BazIO::DeltaCompressionParams&) {
            std::vector<TOut> ret;
            ret.reserve(raw.size());
            BazIO::DeltaCompression d{};
            for (const auto& val : raw)
                ret.push_back(d.Revert<TOut>(val, storeSigned));
            return ret;
        };
        return info.var.Visit(
            std::move(noop),
            std::move(codec),
            std::move(delta),
            [&](const BazIO::FixedPointParams&) -> std::vector<TOut> { throw PBException("Datatype does not support FixedPoint format"); }
        );
    }

    // Deserializes float fields (where things like the FixedPoint transformation
    // are required)
    template <typename TOut>
    std::vector<TOut> ConvertFloat(const std::vector<uint32_t>& raw,
                                   const BazIO::TransformsParams& info,
                                   BazIO::PacketFieldName name,
                                   BazIO::StoreSigned storeSigned)
    {
        static_assert(std::is_floating_point<TOut>::value, "Output type should be integral");

        auto func = [&](const BazIO::FixedPointParams& params) {
            std::vector<TOut> ret;
            ret.reserve(raw.size());
            for (const auto& val : raw)
                ret.push_back(BazIO::FixedPoint::Revert<TOut>(val, storeSigned, static_cast<BazIO::FixedPointScale>(params.scale)));
            return ret;
        };
        return info.var.Visit(
            std::move(func),
            [&](const auto&) -> std::vector<TOut> { throw PBException("Datatype only supports FixedPoint format"); }
        );
    }
}

BazEventData::BazEventData(const Primary::RawEventData& packets)
    : numEvents_(packets.NumEvents())
    , internal_(packets.HasPacketField(BazIO::PacketFieldName::IsBase))
{
    for (const auto& field : packets.FieldInfo())
    {
        // There can be multiple transforms, but each individual transform is required to accept a uint64_t
        // as the input to it's `Revert` function.  Since `Revert` needs to apply things in reverse order
        // here we can undo all but the first transform in a single loop, since `ConvertIntegral` is
        // guaranteed to be the only function we need to call.  The first transform in the vector will have
        // to be done in the switch statement below, where we know from the variable name if we need to
        // do an integral or a floating point conversion
        const auto& raw = [&]()
        {
            if (field.transform.size() == 1)
                return packets.PacketField(field.name);
            else
            {
                auto ret = ConvertIntegral<uint32_t>(packets.PacketField(field.name), field.transform.back(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                if (field.transform.size() > 2)
                    for (size_t i = field.transform.size() - 2; i > 0; --i)
                    {
                        ret = ConvertIntegral<uint32_t>(ret, field.transform[i], field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                    }
                return ret;
            }
        }();
        const auto& isLossy = [&](const auto& transforms)
        {
            auto f = Utility::make_overload([](const CodecParams&) { return true;},
                                            [](const auto&) { return false; });
            for (const auto& t : transforms)
            {
                if (f(t)) return true;
            }
            return false;
        };
        switch(field.name)
        {
        case BazIO::PacketFieldName::Label:
            {
                auto tmp = ConvertIntegral<uint8_t>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                readouts_.reserve(tmp.size());
                static constexpr char tags[4] = {'A', 'C', 'G', 'T'};
                for (auto& r : tmp) readouts_.push_back(tags[r]);
                break;
            }
        case BazIO::PacketFieldName::IsBase:
            {
                isBase_ = ConvertIntegral<bool>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                break;
            }
        case BazIO::PacketFieldName::Pw:
            {
                pws_ = ConvertIntegral<uint32_t>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                break;
            }
        case BazIO::PacketFieldName::StartFrame:
            {
                exactStartFrames_ = !isLossy(field.transform);
                startFrames_ = ConvertIntegral<uint32_t>(raw, field.transform.front(), field.name, field.storeSigned);
                break;
            }
        case BazIO::PacketFieldName::Pkmax:
            {
                pkmax_ = ConvertFloat<float>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                break;
            }
        case BazIO::PacketFieldName::Pkmid:
            {
                pkmid_ = ConvertFloat<float>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                break;
            }
        case BazIO::PacketFieldName::Pkmean:
            {
                pkmean_ = ConvertFloat<float>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                break;
            }
        case BazIO::PacketFieldName::Pkvar:
            {
                pkvar_ = ConvertFloat<float>(raw, field.transform.front(), field.name, static_cast<BazIO::StoreSigned>(field.storeSigned));
                break;
            }
        }
    }

    if (!Internal())
        isBase_ = std::vector<bool>(NumEvents(), true);
}

BazEventData::BazEventData(const std::map<PacketFieldName, std::vector<uint32_t>>& intFields,
                           const std::map<PacketFieldName, std::vector<float>>& floatData,
                           bool exactStartFrames)
    : exactStartFrames_(exactStartFrames)
{
    internal_ = intFields.count(PacketFieldName::IsBase);
    numEvents_ = 0;
    for (const auto& kv : intFields)
    {
        numEvents_ = kv.second.size();
        if (numEvents_) break;
    }
    if (numEvents_ == 0)
    {
        for (const auto& kv : floatData)
        {
            numEvents_ = kv.second.size();
            if (numEvents_) break;
        }
    }

    auto copyVec = [](PacketFieldName field, const auto& map, auto& vec)
    {
        const auto& source = map.at(field);
        vec.reserve(source.size());
        std::copy(source.begin(), source.end(), std::back_inserter(vec));
    };

    auto copyIfPresent = [&](PacketFieldName field, const auto& map, auto& vec)
    {
        if (map.count(field))
        {
            copyVec(field, map, vec);
            return true;
        }
        return false;
    };

    copyVec(PacketFieldName::Label, intFields, readouts_);
    copyVec(PacketFieldName::Pw, intFields, pws_);
    copyVec(PacketFieldName::StartFrame, intFields, startFrames_);

    if(!copyIfPresent(PacketFieldName::IsBase, intFields, isBase_))
    {
        assert(!internal_);
        isBase_ = std::vector<bool>(NumEvents(), true);
    }

    copyIfPresent(PacketFieldName::Pkmid, floatData, pkmid_);
    copyIfPresent(PacketFieldName::Pkmax, floatData, pkmax_);
    copyIfPresent(PacketFieldName::Pkmean, floatData, pkmean_);
    copyIfPresent(PacketFieldName::Pkvar, floatData, pkvar_);

}

}}
