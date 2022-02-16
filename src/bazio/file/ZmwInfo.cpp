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

ZmwInfo::ZmwInfo(const Data& zmwData)
    : zmwData_(zmwData)
{
    std::vector<size_t> vecSizes{ zmwData_.holeNumbers.size(),
                                  zmwData_.holeTypes.size(),
                                  zmwData_.holeX.size(),
                                  zmwData_.holeY.size(),
                                  zmwData_.holeFeaturesMask.size() };
    if(!std::all_of(vecSizes.begin()+1, vecSizes.end(), [&](const size_t s) { return s == vecSizes.front(); }))
        throw PBException("ZmwData does not contain equal sizes for all datasets!");
}

void ZmwInfo::FromJson(const Json::Value& zmwInfo)
{
    zmwData_.holeNumbers = ParseJsonRLEHexArray(zmwInfo, JsonKey::ZmwNumberLut);
    zmwData_.holeFeaturesMask = ParseJsonRLEHexArray(zmwInfo, JsonKey::ZmwFeatureLut);
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

    if (zmwInfo.isMember(JsonKey::ZmwFeatureMap))
    {
        if (!DataSource::ZmwFeatures::validateCsvMapString(zmwInfo[JsonKey::ZmwFeatureMap].asString()))
        {
            throw PBException("Incompatible hole features map with current ZMW features!");
        }
    }
    else
    {
        throw PBException("Missing " + std::string(JsonKey::ZmwFeatureMap) + " in BAZ header!");
    }
}

Json::Value ZmwInfo::ToJson() const
{
    Json::Value zmwInfo;
    zmwInfo[JsonKey::ZmwNumberLut] = ZmwNumberLut();
    zmwInfo[JsonKey::ZmwTypeLut] = ZmwTypeLut();
    zmwInfo[JsonKey::ZmwXYLut] = ZmwXYLut();
    zmwInfo[JsonKey::ZmwFeatureLut] = ZmwFeatureLut();
    zmwInfo[JsonKey::ZmwFeatureMap] = DataSource::ZmwFeatures::csvMapString();
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

Json::Value ZmwInfo::ZmwFeatureLut() const
{
    return RunLengthEncLUTHexJson(RunLengthEncLUT(zmwData_.holeFeaturesMask));
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

} // PacBio::BazIO

