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

#include <bazio/file/RunLength.h>

using namespace PacBio::BazIO;

TEST(RunLength, Encode)
{
    std::vector<size_t> startNum{10, 100, 150, 1000};
    std::vector<uint32_t> vecSizes{10, 15, 1, 25};

    std::vector<uint32_t> l;
    for (size_t i = 0; i < vecSizes.size(); i++)
    {
        std::vector<uint32_t> m(vecSizes[i]);
        std::iota(m.begin(), m.end(), startNum[i]);
        l.insert(l.end(), m.begin(), m.end());
    }

    const auto& enc = RunLengthEncLUT(l);
    EXPECT_TRUE(enc.size() == vecSizes.size());
    for (size_t i = 0; i < enc.size(); i++)
    {
        EXPECT_TRUE(enc[static_cast<int>(i)].first == startNum[i] && enc[static_cast<int>(i)].second == vecSizes[i]);
    }

    const auto& json = RunLengthEncLUTHexJson(enc);
    EXPECT_TRUE(json.isArray());
    EXPECT_TRUE(json.size() == vecSizes.size());

    std::vector<uint32_t> s;
    for (size_t i = 0; i < vecSizes.size(); i++)
    {
        std::vector<uint32_t> m(vecSizes[i], startNum[i]);
        s.insert(s.end(), m.begin(), m.end());
    }

    const auto& senc = RunLengthSameEncLUT(s);
    for (size_t i = 0; i < senc.size(); i++)
    {
        EXPECT_TRUE(senc[static_cast<int>(i)].first == startNum[i] && senc[static_cast<int>(i)].second == vecSizes[i]);
    }

    const auto& tenc = RunLengthSameEncLUT(l);
    EXPECT_TRUE(tenc.size() == std::accumulate(vecSizes.begin(), vecSizes.end(), 0u));

}

TEST(RunLength, Decode)
{
    std::vector<size_t> startNum{10, 100, 150, 1000};
    std::vector<uint32_t> vecSizes{10, 15, 1, 25};

    Json::Value val;
    for (size_t i = 0; i < vecSizes.size(); i++)
    {
        Json::Value v;
        std::stringstream ss;
        ss << "0x" << std::hex << startNum[i] << std::dec;
        v.append(ss.str());
        v.append(vecSizes[i]);
        val.append(std::move(v));
    }

    const auto& l = RunLengthDecLUTHexJson(val);
    EXPECT_TRUE(l.size() == std::accumulate(vecSizes.begin(), vecSizes.end(), 0u));
    EXPECT_TRUE(l[0] == startNum[0]);
    EXPECT_TRUE(l[10] == startNum[1]);
    EXPECT_TRUE(l[25] == startNum[2]);
    EXPECT_TRUE(l[26] == startNum[3]);

    const auto& m = RunLengthSameDecLUTHexJson(val);
    EXPECT_TRUE(m.size() == std::accumulate(vecSizes.begin(), vecSizes.end(), 0u));
    EXPECT_TRUE(std::all_of(m.begin(), m.begin() + 10, [](const uint32_t v) { return v == 10; }));
    EXPECT_TRUE(std::all_of(m.begin() + 10, m.begin() + 10 + 15 - 1, [](const uint32_t v) { return v == 100; }));
    EXPECT_TRUE(std::all_of(m.begin() + 10 + 15, m.begin() + 10 + 15, [](const uint32_t v) { return v == 150; }));
    EXPECT_TRUE(std::all_of(m.begin() + 10 + 15 + 1, m.end(), [](const uint32_t v) { return v == 1000; }));
}