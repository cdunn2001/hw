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

#include <numeric>

#include <gtest/gtest.h>

#include <bazio/file/ZmwInfo.h>

using namespace PacBio::BazIO;

ZmwInfo MakeZmwInfo(size_t numZmws, uint8_t setHoleType, uint32_t setHoleFeature)
{
    std::vector<uint32_t> zmwNumbers(numZmws);
    std::vector<uint16_t> zmwX(numZmws);
    std::iota(zmwNumbers.begin(), zmwNumbers.end(), 0);
    std::iota(zmwX.begin(), zmwX.end(), 0);

    ZmwInfo zi(ZmwInfo::Data{ zmwNumbers,
                              std::vector<uint8_t>(zmwNumbers.size(), setHoleType),
                              zmwX,
                              zmwX,
                              std::vector<uint32_t>(zmwNumbers.size(), setHoleFeature)
    });

    return zi;
}

TEST(ZmwInfo, Construct)
{
    size_t numZmws = 10000;
    uint8_t setHoleType = 1;
    uint32_t setHoleFeature = 0;

    ZmwInfo zi = MakeZmwInfo(numZmws, setHoleType, setHoleFeature);

    EXPECT_EQ(numZmws, zi.NumZmws());

    const auto& hns = zi.HoleNumbers();
    for (size_t i = 0; i < hns.size(); i++)
        EXPECT_EQ(i, hns[i]);

   const auto& zmwXY = zi.HoleXY();
   for (size_t i = 0; i < zmwXY.size(); i++)
   {
       EXPECT_EQ(i, zmwXY[i].first);
       EXPECT_EQ(i, zmwXY[i].second);
   }

   const auto& zmwTypes = zi.HoleTypes();
   EXPECT_TRUE(std::all_of(zmwTypes.begin(), zmwTypes.end(),
               [&setHoleType](const uint8_t ht) { return ht == setHoleType; }));

   const auto& zmwFeatures = zi.UnitFeatures();
   EXPECT_TRUE(std::all_of(zmwFeatures.begin(), zmwFeatures.end(),
                            [&setHoleFeature](const uint32_t hf) { return hf == setHoleFeature; }));
}

TEST(ZmwInfo, Deserialize)
{
    size_t numZmws = 10000;
    uint8_t setHoleType = 1;
    uint32_t setHoleFeature = 0;

    ZmwInfo zi1 = MakeZmwInfo(numZmws, setHoleType, setHoleFeature);

    ZmwInfo zi2;
    zi2.FromJson(zi1.ToJson());

    const auto& hns = zi2.HoleNumbers();
    for (size_t i = 0; i < hns.size(); i++)
        EXPECT_EQ(i, hns[i]);

    const auto& zmwXY = zi2.HoleXY();
    for (size_t i = 0; i < zmwXY.size(); i++)
    {
        EXPECT_EQ(i, zmwXY[i].first);
        EXPECT_EQ(i, zmwXY[i].second);
    }
}