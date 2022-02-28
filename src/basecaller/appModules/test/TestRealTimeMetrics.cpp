// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#include <gtest/gtest.h>

#include <pacbio/datasource/ZmwFeatures.h>
#include <common/simd/SimdConvTraits.h>
#include <common/MongoConstants.h>

#include <appModules/RealTimeMetrics.h>

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo;
using namespace PacBio::Simd;

TEST(RealTimeMetrics, SelectedLanesWithFeatures)
{
    RealTimeMetrics rtm;

    size_t numLanes = 4;
    size_t numZmws = numLanes * laneSize;
    std::vector<uint32_t> features(numLanes * laneSize, static_cast<uint32_t>(ZmwFeatures::Sequencing));

    for (size_t i = 0; i < numZmws; i++)
    {
        if (i % 2) features[i] = static_cast<uint32_t>(ZmwFeatures::LaserScatter);
    }

    auto lms = rtm.SelectedLanesWithFeatures(features, static_cast<uint32_t>(ZmwFeatures::Sequencing));

    for (const auto& lm : lms)
    {
        for (size_t i = 0; i < laneSize; i++)
        {
            if (!(i % 2)) EXPECT_TRUE(lm[i]);
            else EXPECT_FALSE(lm[i]);
        }
    }

    for (size_t i = 0; i < numZmws; i++)
    {
        if (i % 3 == 0) features[i] = static_cast<uint32_t>(ZmwFeatures::Sequencing);
        if (i % 3 == 1) features[i] = static_cast<uint32_t>(ZmwFeatures::Sequencing)
                                    | static_cast<uint32_t>(ZmwFeatures::LaserScatter);
        if (i % 3 == 2) features[i] = static_cast<uint32_t>(ZmwFeatures::Sequencing)
                                    | static_cast<uint32_t>(ZmwFeatures::LaserPower2p0x);
    }

    lms = rtm.SelectedLanesWithFeatures(features, static_cast<uint32_t>(ZmwFeatures::Sequencing));
    for (const auto& lm : lms)
    {
        for (size_t i = 0; i < laneSize; i++)
        {
            EXPECT_TRUE(lm[i]);
        }
    }

    lms = rtm.SelectedLanesWithFeatures(features, static_cast<uint32_t>(ZmwFeatures::LaserScatter));
    size_t zmwNo = 0;
    for (const auto& lm : lms)
    {
        for (size_t i = 0; i < laneSize; i++)
        {
            if (zmwNo % 3 == 1) EXPECT_TRUE(lm[i]);
            else EXPECT_FALSE(lm[i]);
            zmwNo++;
        }
    }

    lms = rtm.SelectedLanesWithFeatures(features, static_cast<uint32_t>(ZmwFeatures::LaserPower2p0x));
    zmwNo = 0;
    for (const auto& lm : lms)
    {
        for (size_t i = 0; i < laneSize; i++)
        {
            if (zmwNo % 3 == 2) EXPECT_TRUE(lm[i]);
            else EXPECT_FALSE(lm[i]);
            zmwNo++;
        }
    }
}
