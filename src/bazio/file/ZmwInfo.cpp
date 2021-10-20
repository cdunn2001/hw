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

#include <pacbio/PBException.h>

#include "ZmwInfo.h"

namespace PacBio::BazIO
{

ZmwInfo::ZmwInfo(const Data& zmwData,
                 const std::map<std::string, uint32_t>& holeTypesMap,
                 const std::map<std::string, uint32_t>& holeFeaturesMap)
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

void ZmwInfo::FromJson(const Json::Value& zmwInfo)
{
    zmwData_.holeNumbers = ParseJsonRLEHexArray(zmwInfo, "ZMW_NUMBER_LUT");
    zmwData_.holeFeatures = ParseJsonRLEHexArray(zmwInfo, "ZMW_UNIT_FEATURE_LUT");
    std::vector<uint32_t> holeTypes = ParseJsonRLEHexArray(zmwInfo, "ZMW_TYPE_LUT");
    std::vector<uint32_t> holeX = ParseJsonRLEHexArray(zmwInfo["ZMW_XY_LUT"], "X");
    std::vector<uint32_t> holeY = ParseJsonRLEHexArray(zmwInfo["ZMW_XY_LUT"], "Y");

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

    holeTypesMap_ = ParseJsonMap(zmwInfo, "ZMW_TYPE_MAP");
    holeFeaturesMap_ = ParseJsonMap(zmwInfo, "ZMW_UNIT_FEATURE_MAP");
}

Json::Value ZmwInfo::ToJson() const
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

std::map<std::string, uint32_t> ZmwInfo::ParseJsonMap(const Json::Value& node, const std::string& field) const
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

Json::Value ZmwInfo::EncodeMapJson(const std::map<std::string, uint32_t>& map) const
{
    Json::Value jsonMap;
    for (const auto& [k,v] : map)
    {
        jsonMap[k] = v;
    }
    return jsonMap;
}

} // PacBio::BazIO

