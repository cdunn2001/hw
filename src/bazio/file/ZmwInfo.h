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
#include <unordered_map>

#include <json/json.h>

#include "RunLength.h"

namespace PacBio::BazIO
{

/// This class is meant to store the following information for each ZMW:
///
/// HoleNumber          ZMW hole number
/// HoleType            These are specific to a particular chip layout
/// HoleXY              X-Y coordinates of ZMW with respect to the chip
/// HoleFeaturesMask    Bitmask of various physical features of the ZMW
///

class ZmwInfo
{
public:

    class JsonKey
    {
    public:
        // JSON keys for the various datasets
        static constexpr auto ZmwInfo = "ZMW_INFO";
        static constexpr auto ZmwNumberLut = "ZMW_NUMBER_LUT";
        static constexpr auto ZmwFeatureLut = "ZMW_FEATURE_LUT";
        static constexpr auto ZmwTypeLut = "ZMW_TYPE_LUT";
        static constexpr auto ZmwXYLut = "ZMW_XY_LUT";
        static constexpr auto ZmwX = "X";
        static constexpr auto ZmwY = "Y";
        static constexpr auto ZmwFeatureMap = "ZMW_FEATURE_MAP";
    };

public:

    // Simple struct to package up the various vectors of data
    struct Data
    {
        Data() = default;

        Data(const std::vector<uint32_t>& hns, const std::vector<uint8_t>& hts,
             const std::vector<uint16_t>& hxs, const std::vector<uint16_t>& hys,
             const std::vector<uint32_t>& hfms)
            : holeNumbers(hns)
            , holeTypes(hts)
            , holeX(hxs)
            , holeY(hys)
            , holeFeaturesMask(hfms)
        { }

        std::vector<uint32_t>   holeNumbers;
        std::vector<uint8_t>    holeTypes;
        std::vector<uint16_t>   holeX;
        std::vector<uint16_t>   holeY;
        std::vector<uint32_t>   holeFeaturesMask;
    };

public:

    ZmwInfo(const Data& zmwData);

    ZmwInfo() = default;
    ZmwInfo(ZmwInfo&&) = default;
    ZmwInfo(const ZmwInfo&) = default;
    ZmwInfo& operator=(ZmwInfo&&) = default;
    ZmwInfo& operator=(const ZmwInfo&) = default;
    ~ZmwInfo() = default;

public:

    static ZmwInfo FromJson(const Json::Value& zmwInfo);
    static ZmwInfo CombineInfos(const std::vector<std::reference_wrapper<const ZmwInfo>>& infos);

public:

    const std::vector<uint32_t>& HoleNumbers() const
    { return zmwData_.holeNumbers; }

    const std::vector<uint8_t>& HoleTypes() const
    { return zmwData_.holeTypes; }

    const std::vector<uint16_t>& HoleX() const
    { return zmwData_.holeX; }

    const std::vector<uint16_t>& HoleY() const
    { return zmwData_.holeY; }

    const std::vector<uint32_t>& HoleFeaturesMask() const
    { return zmwData_.holeFeaturesMask; }

    const std::unordered_map<uint32_t,uint32_t>& ZmwNumbersToIndex() const
    { return zmwNumbersToIndex_; }

    uint32_t ZmwIndexToNumber(const uint32_t index) const
    { return zmwData_.holeNumbers.at(index); }

    uint32_t ZmwNumberToIndex(const uint32_t holeNumber) const
    { return zmwNumbersToIndex_.at(holeNumber); }

    size_t NumZmws() const
    { return zmwData_.holeNumbers.size(); }

public:

    // Encoding to JSON methods
    Json::Value ToJson() const;
    Json::Value ZmwNumberLut() const;
    Json::Value ZmwTypeLut() const;
    Json::Value ZmwXYLut() const;
    Json::Value ZmwFeatureLut() const;

private:

    static std::vector<uint32_t> ParseJsonRLEHexArray(const Json::Value& node, const std::string& field);
    static std::vector<uint32_t> ParseJsonRLESameHexArray(const Json::Value& node, const std::string& field);

private:
    Data                            zmwData_;
    std::unordered_map<uint32_t,uint32_t>     zmwNumbersToIndex_;
};


} // PacBio::BazIO

#endif // BAZIO_FILE_ZMW_INFO_H
