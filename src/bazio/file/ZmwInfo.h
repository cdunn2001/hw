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

#ifndef BAZIO_FILE_ZMW_INFO_H
#define BAZIO_FILE_ZMW_INFO_H

#include <algorithm>
#include <functional>

#include <json/json.h>

namespace PacBio::BazIO
{

/// This class is meant to store the following information for each ZMW:
///
/// HoleNumber
/// HoleType        These are specific to a particular chip layout
/// HoleXY          X-Y coordinates of ZMW with respect to the chip
/// UnitFeature     Bitmask of various physical features of the ZMW
///

class ZmwInfo
{
public:

    static Json::Value RunLengthEncLUTJson(const std::vector<std::pair<uint32_t, uint32_t>>& input);
    static Json::Value RunLengthEncLUTHexJson(const std::vector<std::pair<uint32_t, uint32_t>>& input);
    static std::vector<std::pair<uint32_t, uint32_t>> RunLengthEncLUT(const std::vector<uint32_t>& input);

public:
    struct Data
    {
        uint32_t    holeNumber;
        uint8_t     holeType;
        uint16_t    holeX;
        uint16_t    holeY;
        uint32_t    holeFeature;
    };

public:
    ZmwInfo(const std::vector<Data>& zmwData,
            const std::map<std::string, uint32_t>& holeTypesMap,
            const std::map<std::string, uint32_t>& holeFeaturesMap)
        : zmwData_(zmwData)
        , holeTypesMap_(holeTypesMap)
        , holeFeaturesMap_(holeFeaturesMap)
    { }

    ZmwInfo(const Json::Value& zmwInfo)
    {
        std::vector<uint32_t> holeNumbers = ParseJsonRLEHexArray(zmwInfo, "ZMW_NUMBER_LUT");
        std::vector<uint32_t> holeType = ParseJsonRLEHexArray(zmwInfo, "ZMW_TYPE_LUT");
        std::vector<uint32_t> holeX = ParseJsonRLEHexArray(zmwInfo["ZMW_XY_LUT"], "X");
        std::vector<uint32_t> holeY = ParseJsonRLEHexArray(zmwInfo["ZMW_XY_LUT"], "Y");
        std::vector<uint32_t> holeFeatures = ParseJsonRLEHexArray(zmwInfo, "ZMW_UNIT_FEATURE_LUT");

        for (size_t i = 0; i < holeNumbers.size(); i++)
        {
            zmwData_.emplace_back(Data{ holeNumbers[i],
                                        static_cast<uint8_t>(holeType[i]),
                                        static_cast<uint16_t>(holeX[i]), static_cast<uint16_t>(holeY[i]),
                                        holeFeatures[i] });
            zmwNumbersToId_[holeNumbers[i]] = i;
        }

        holeTypesMap_ = ParseJsonMap(zmwInfo, "ZMW_TYPE_MAP");
        holeFeaturesMap_ = ParseJsonMap(zmwInfo, "ZMW_UNIT_FEATURE_MAP");
    }

public:
    const std::vector<Data>& ZmwData() const
    { return zmwData_; }

    const std::map<std::string,uint32_t>& HoleTypesMap() const
    { return holeTypesMap_; }

    const std::map<std::string,uint32_t>& HoleFeatureMap() const
    { return holeFeaturesMap_; }

    uint32_t ZmwIdToNumber(const uint32_t id) const
    { return zmwData_.at(id).holeNumber; }

public:
    Json::Value ToJson() const
    {
        Json::Value zmwInfo;
        zmwInfo["ZMW_NUMBER_LUT"] = ZmwNumberLut();
        zmwInfo["ZMW_TYPE_LUT"] = ZmwTypeLut();
        zmwInfo["ZMW_XY_LUT"] = ZmwXYLut();
        zmwInfo["ZMW_UNIT_FEATURE_LUT"] = ZmwUnitFeatureLut();
        zmwInfo["ZMW_TYPE_MAP"] = ZmwHoleTypeMap();
        zmwInfo["ZMW_UNIT_FEATURE_MAP"] = ZmwFeatureTypeMap();
        return zmwInfo;
    }

    Json::Value ZmwNumberLut() const
    {
        return EncodeJson([](const Data& d) { return static_cast<uint32_t>(d.holeNumber); });
    }

    Json::Value ZmwTypeLut() const
    {
        return EncodeJson([](const Data& d) { return static_cast<uint32_t>(d.holeType); });
    }

    Json::Value ZmwXYLut() const
    {
        Json::Value holeXY;
        holeXY["X"] = EncodeJson([](const Data& d) { return static_cast<uint32_t>(d.holeX); });
        holeXY["Y"] = EncodeJson([](const Data& d) { return static_cast<uint32_t>(d.holeY); });
        return holeXY;
    }

    Json::Value ZmwUnitFeatureLut() const
    {
        return EncodeJson([](const Data& d) { return static_cast<uint32_t>(d.holeFeature); });
    }

    Json::Value ZmwHoleTypeMap() const
    {
        return EncodeMapJson(holeTypesMap_);
    }

    Json::Value ZmwFeatureTypeMap() const
    {
        return EncodeMapJson(holeFeaturesMap_);
    }

private:

    std::vector<uint32_t> ParseJsonRLEHexArray(const Json::Value& node, const std::string& field) const
    {
        std::vector<uint32_t> data;
        if (node.isMember(field) && node[field].isArray())
        {
            Json::Value vals = node[field];
            for (size_t i = 0; i < vals.size(); ++i)
            {
                Json::Value singleZmw = vals[static_cast<int>(i)];
                uint32_t start = std::stoul(singleZmw[0].asString(), nullptr, 16);
                uint32_t runLength = singleZmw[1].asUInt();
                for (uint32_t j = 0; j < runLength; ++j)
                {
                    data.emplace_back(start + j);
                }
            }
        }
        return data;
    }

    std::map<std::string, uint32_t> ParseJsonMap(const Json::Value& node, const std::string& field) const
    {
        std::map<std::string,uint32_t> data;
        if (node.isMember(field) && node[field].isObject())
        {
            for (const auto& id : node[field].getMemberNames())
            {
                data[id] = node[field][id].asUInt();
            }
        }
        return data;
    }

    Json::Value EncodeJson(const std::function<uint32_t(const Data& d)>& extract) const
    {
        std::vector<uint32_t> holeInfo;
        holeInfo.resize(zmwData_.size());
        std::transform(zmwData_.begin(), zmwData_.end(), std::back_inserter(holeInfo),
                       [&extract](const Data& d) { return extract(d); });
        return RunLengthEncLUTHexJson(RunLengthEncLUT(holeInfo));
    }

    Json::Value EncodeMapJson(const std::map<std::string, uint32_t>& map) const
    {
        Json::Value jsonMap;
        for (const auto& [k,v] : map)
        {
            jsonMap[k] = v;
        }
        return jsonMap;
    }

    std::vector<Data>      zmwData_;
    std::map<uint32_t,uint32_t>     zmwNumbersToId_;
    std::map<std::string, uint32_t> holeTypesMap_;
    std::map<std::string, uint32_t> holeFeaturesMap_;
};


} // PacBio::BazIO

#endif // BAZIO_FILE_ZMW_INFO_H
