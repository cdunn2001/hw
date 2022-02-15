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

#include <sstream>

#include <pacbio/datasource/ZmwFeatures.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/text/String.h>
#include <pacbio/PBException.h>


#include "ZmwInfo.h"

namespace PacBio::BazIO
{



ZmwInfo::ZmwInfo(const Data& zmwData,
                 const std::string& holeTypesMap,
                 const std::string& holeFeaturesMap)
    : zmwData_(zmwData)
    , holeTypesMap_(holeTypesMap)
    , holeFeaturesMap_(holeFeaturesMap)
{
    std::vector<size_t> vecSizes{ zmwData_.holeNumbers.size(),
                                  zmwData_.holeTypes.size(),
                                  zmwData_.holeX.size(),
                                  zmwData_.holeY.size(),
                                  zmwData_.holeFeatures.size() };
    if(!std::all_of(vecSizes.begin()+1, vecSizes.end(), [&](const size_t s) { return s == vecSizes.front(); }))
        throw PBException("ZmwData does not contain equal sizes for all datasets!");
}

ZmwInfo::ZmwInfo(const Data& zmwData)
    : ZmwInfo(zmwData, "1=SEQUENCING", DataSource::ZmwFeatures::csvMapString())
{
    PBLOG_WARN << "Default ZmwInfo unit types currently mocked out";
}

void ZmwInfo::FromJson(const Json::Value& zmwInfo)
{
    zmwData_.holeNumbers = ParseJsonRLEHexArray(zmwInfo, JsonKey::ZmwNumberLut);
    zmwData_.holeFeatures = ParseJsonRLEHexArray(zmwInfo, JsonKey::ZmwUnitFeatureLut);
    std::vector<uint32_t> holeTypes = ParseJsonRLEHexArray(zmwInfo, JsonKey::ZmwTypeLut);
    std::vector<uint32_t> holeX = ParseJsonRLEHexArray(zmwInfo[JsonKey::ZmwXYLut], JsonKey::ZmwX);
    std::vector<uint32_t> holeY = ParseJsonRLEHexArray(zmwInfo[JsonKey::ZmwXYLut], JsonKey::ZmwY);

    zmwData_.holeTypes.resize(NumZmws());
    zmwData_.holeX.resize(NumZmws());
    zmwData_.holeY.resize(NumZmws());
    for (size_t i = 0; i < NumZmws(); i++)
    {
        zmwData_.holeTypes[i] = static_cast<uint8_t>(holeTypes[i]);
        zmwData_.holeX[i] = static_cast<uint16_t>(holeX[i]);
        zmwData_.holeY[i] = static_cast<uint16_t>(holeY[i]);
        zmwNumbersToIndex_[zmwData_.holeNumbers[i]] = i;
    }

    holeTypesMap_ = ParseJsonMap(zmwInfo, JsonKey::ZmwTypeMap);
    holeFeaturesMap_ = ParseJsonMap(zmwInfo, JsonKey::ZmwUnitFeatureMap);
    if (!DataSource::ZmwFeatures::validateCsvMapString(holeFeaturesMap_))
    {
        throw PBException("Incompatible hole features map with current ZMW features!");
    }
}

Json::Value ZmwInfo::ToJson() const
{
    Json::Value zmwInfo;
    zmwInfo[JsonKey::ZmwNumberLut] = ZmwNumberLut();
    zmwInfo[JsonKey::ZmwTypeLut] = ZmwTypeLut();
    zmwInfo[JsonKey::ZmwXYLut] = ZmwXYLut();
    zmwInfo[JsonKey::ZmwUnitFeatureLut] = ZmwUnitFeatureLut();
    zmwInfo[JsonKey::ZmwTypeMap] = ZmwHoleTypeMap();
    zmwInfo[JsonKey::ZmwUnitFeatureMap] = ZmwFeatureTypeMap();
    return zmwInfo;
}

Json::Value ZmwInfo::ZmwNumberLut() const
{
    return RunLengthEncLUTHexJson(RunLengthEncLUT(zmwData_.holeNumbers));
}

Json::Value ZmwInfo::ZmwTypeLut() const
{
    return RunLengthEncLUTHexJson(RunLengthEncLUT(zmwData_.holeTypes));
}

Json::Value ZmwInfo::ZmwXYLut() const
{
    Json::Value holeXY;
    // TODO: Depending upon how the hole numbers are specified and if they are in
    // readout order than run length encoding of the X-coordinate (major dimension)
    // is wasteful since the Y-coordinate will be increasing and instead the X-coordinate
    // should be encoded using runs consisting of the same value.
    holeXY["X"] = RunLengthEncLUTHexJson(RunLengthEncLUT(zmwData_.holeX));
    holeXY["Y"] = RunLengthEncLUTHexJson(RunLengthEncLUT(zmwData_.holeY));
    return holeXY;
}

Json::Value ZmwInfo::ZmwUnitFeatureLut() const
{
    return RunLengthEncLUTHexJson(RunLengthEncLUT(zmwData_.holeFeatures));
}

Json::Value ZmwInfo::ZmwHoleTypeMap() const
{
    return EncodeMapJson(holeTypesMap_);
}

Json::Value ZmwInfo::ZmwFeatureTypeMap() const
{
    return EncodeMapJson(holeFeaturesMap_);
}

std::vector<uint32_t> ZmwInfo::ParseJsonRLEHexArray(const Json::Value& node, const std::string& field) const
{
    if (node.isMember(field))
    {
        return RunLengthDecLUTHexJson(node[field]);
    }
    else
    {
        throw PBException("BAZ header doesn't contain field: " + field);
    }
}

std::string ZmwInfo::ParseJsonMap(const Json::Value& node, const std::string& field) const
{
    auto cmp = [](const std::string& a, const std::string& b) {
        return
            atoi(PacBio::Text::String::Split(a, '=').front().c_str()) <
            atoi(PacBio::Text::String::Split(b, '=').front().c_str());
    };
    std::set<std::string, decltype(cmp)> data(cmp);
    if (node.isMember(field) && node[field].isObject())
    {
        for (const auto& id: node[field].getMemberNames())
        {
            data.insert(id + "=" + node[field][id].asString());
        }
    }
    return PacBio::Text::String::Join(data.begin(), data.end(), ',');
}

Json::Value ZmwInfo::EncodeMapJson(const std::string& strMap) const
{
    Json::Value jsonMap;
    for (const auto& v: PacBio::Text::String::Split(strMap, ','))
    {
        const auto& vals = PacBio::Text::String::Split(v, '=');
        jsonMap[vals.front()] = vals.back();
    }
    return jsonMap;
}

} // PacBio::BazIO

