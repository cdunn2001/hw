// Copyright (c) 2018-2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_BAZIO_BAZ_EVENT_DATA_H_
#define PACBIO_BAZIO_BAZ_EVENT_DATA_H_

#include <bazio/DataParsing.h>
#include <bazio/encoding/FieldNames.h>

namespace PacBio {
namespace BazIO {

// Provides read-only access to event data extracted from the baz file.
// It has been fully deserialzed and all transformations reverted.  This
// class does not perform any significant post-processing of whatever was
// stored in the baz file.  With the exception of making sure the IsBase
// vector is populated, nothing will be present in this object that was not
// directly serialized in the baz file itself
class BazEventData
{
public:
    BazEventData(const std::map<PacketFieldName, std::vector<uint32_t>>& intFields,
                 const std::map<PacketFieldName, std::vector<float>>& floatData,
                 // Not sure what to assume here, this seems conservative
                 bool exactStartFrames = false)
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

    explicit BazEventData(const Primary::RawEventData& packets);

    // Move only semantics
    BazEventData(const BazEventData&) = delete;
    BazEventData(BazEventData&&) = default;
    BazEventData& operator=(const BazEventData&) = delete;
    BazEventData& operator=(BazEventData&&) = default;

public: // const data accesors
    const std::vector<bool>& IsBase() const { return isBase_; }
    bool IsBase(size_t idx) const { return isBase_[idx]; }

    const std::vector<uint32_t>& PulseWidths() const { return pws_; }
    uint32_t PulseWidths(size_t idx) const { return pws_[idx]; }

    const std::vector<char>& Readouts() const
    { return readouts_; }

    const std::vector<uint32_t>& StartFrames() const
    { return startFrames_; }

    const std::vector<float>& PkMeans() const { return pkmean_; }

    const std::vector<float>& PkMids() const { return pkmid_; }

    size_t NumEvents() const { return numEvents_; }
    bool Internal() const { return internal_; }
    bool StartFramesAreExact() const { return exactStartFrames_; }

    Json::Value EventToJson(size_t idx) const
    {
        Json::Value ret;
        for (auto field : PacketFieldName::allValues())
        {
            const std::string name = field.toString();
            switch(field)
            {
            case PacketFieldName::IsBase:
                {
                    if (!isBase_.empty()) ret[name] = isBase_[idx];
                    break;
                }
            case PacketFieldName::Label:
                {
                    if (!readouts_.empty()) ret[name] = std::string(1, readouts_[idx]);
                    break;
                }
            case PacketFieldName::Pw:
                {
                    if (!pws_.empty()) ret[name] = pws_[idx];
                    break;
                }
            case PacketFieldName::StartFrame:
                {
                    if (!startFrames_.empty()) ret[name] = startFrames_[idx];
                    break;
                }
            case PacketFieldName::Pkmid:
                {
                    if (!pkmid_.empty()) ret[name] = pkmid_[idx];
                    break;
                }
            case PacketFieldName::Pkmax:
                {
                    if (!pkmax_.empty()) ret[name] = pkmax_[idx];
                    break;
                }
            case PacketFieldName::Pkmean:
                {
                    if (!pkmean_.empty()) ret[name] = pkmean_[idx];
                    break;
                }
            case PacketFieldName::Pkvar:
                {
                    if (!pkvar_.empty()) ret[name] = pkvar_[idx];
                    break;
                }
            }
        }
        return ret;
    }

private:
    size_t numEvents_;
    bool internal_;
    bool exactStartFrames_;

    std::vector<bool> isBase_;
    std::vector<char> readouts_;
    std::vector<uint32_t> pws_;
    std::vector<uint32_t> startFrames_;

    std::vector<float> pkmid_;
    std::vector<float> pkmax_;
    std::vector<float> pkmean_;
    std::vector<float> pkvar_;
};

}} // ::PacBio::Primary

#endif //PACBIO_BAZIO_BAZ_EVENT_DATA_H_
