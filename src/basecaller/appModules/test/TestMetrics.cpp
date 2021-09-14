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

#include <gtest/gtest.h>

#include <appModules/Metrics.h>

using namespace PacBio::Application;
using namespace PacBio::Mongo::Data;

BasecallingMetrics MakeBasecallingMetrics()
{
    BasecallingMetrics bm;
    bm.activityLabel = HQRFPhysicalStates::SINGLE;
    bm.numPulseFrames = 100;
    bm.numBaseFrames = 100;
    bm.numSandwiches = 10;
    bm.numHalfSandwiches = 20;
    bm.numPulseLabelStutters = 30;
    bm.numBases = 20;
    bm.numPulses = 30;

    bm.numPkMidFrames = BasecallingMetrics::SingleMetric<uint16_t>{300};
    bm.pkMidSignal = BasecallingMetrics::SingleMetric<float>{238.18f * 300};
    bm.bpZvar = BasecallingMetrics::SingleMetric<float>{3.32f};
    bm.pkZvar = BasecallingMetrics::SingleMetric<float>{4.46f};
    bm.pkMax = BasecallingMetrics::SingleMetric<float>{347.9f};
    bm.numPkMidBasesByAnalog = BasecallingMetrics::SingleMetric<uint16_t>{200};

    bm.frameBaselineDWS = 140.0f;
    bm.frameBaselineVarianceDWS = 35.9f;
    bm.numFramesBaseline = 200;
    bm.numFrames = 4096;
    bm.autocorrelation = 10.32f;
    bm.pulseDetectionScore = 5.6f;

    return bm;
}

TEST(Metrics, Production)
{
    using MetricT = ProductionMetrics<uint16_t,float>;
    const auto& bm = MakeBasecallingMetrics();

    MetricT m1{bm, 0};
    MetricT m2(bm, 1);
    m1.Aggregate(m2);

    EXPECT_EQ(bm.numFrames[0]*2, m1.numFrames);
    EXPECT_EQ(bm.numPulseFrames[0]*2, m1.pulseWidth);
    EXPECT_EQ(bm.numBaseFrames[0]*2, m1.baseWidth);

    EXPECT_EQ(bm.numPulses[0]*2, m1.numPulses);
    EXPECT_EQ(bm.numBases[0]*2, m1.numBases);
    EXPECT_FLOAT_EQ( bm.pkMidSignal[0][0]*2, m1.pkmid[0]*m1.numPkmidFrames[0]);
    EXPECT_EQ(bm.numPkMidFrames[0][0]*2, m1.numPkmidFrames[0]);
    EXPECT_FLOAT_EQ(bm.frameBaselineDWS[0], m1.baselineMean);
    EXPECT_FLOAT_EQ(bm.frameBaselineVarianceDWS[0], m1.baselineVar);
    EXPECT_EQ(bm.numFramesBaseline[0]*2, m1.numBaselineFrames);

    EXPECT_EQ(static_cast<uint8_t>(HQRFPhysicalStates::SINGLE), m1.ActivityLabel());
}

TEST(Metrics, Internal)
{
    using MetricT = InternalMetrics<uint16_t,float>;
    const auto& bm = MakeBasecallingMetrics();

    MetricT bmb1{bm, 0};
    MetricT bmb2(bm, 1);
    bmb1.Aggregate(bmb2);

    EXPECT_EQ(bm.numSandwiches[0]*2, bmb1.numSandwiches);
    EXPECT_EQ(bm.numHalfSandwiches[0]*2, bmb1.numHalfSandwiches);
    EXPECT_EQ(bm.numPulseLabelStutters[0]*2, bmb1.numPulseLabelStutters);
    EXPECT_FLOAT_EQ(bm.pulseDetectionScore[0], bmb1.pulseDetectionScore);
    EXPECT_FLOAT_EQ(bm.autocorrelation[0], bmb1.traceAutoCorrelation);
    EXPECT_FLOAT_EQ(bm.pkMax[0][0], bmb1.pkmax[0]);
    EXPECT_EQ(bm.numPkMidBasesByAnalog[0][0]*2, bmb1.numPkmidBases[0]);

    EXPECT_FLOAT_EQ(bm.pkZvar[0][0], bmb1.pkzvar[0]);
    EXPECT_FLOAT_EQ(bm.bpZvar[0][0], bmb1.bpzvar[0]);
}

TEST(Metrics, Aggregate)
{
    using MetricT = ProductionMetrics<uint16_t,float>;
    using MetricAggregatedT = ProductionMetrics<uint32_t,float>;
    const auto& bm = MakeBasecallingMetrics();

    MetricAggregatedT bmb1;
    MetricT bmb2{bm, 0};
    MetricT bmb3{bm, 1};

    bmb1.Aggregate(bmb2);
    bmb1.Aggregate(bmb3);

    EXPECT_EQ(bmb2.numFrames + bmb3.numFrames, bmb1.numFrames);
    EXPECT_EQ(bmb2.pulseWidth + bmb2.pulseWidth, bmb1.pulseWidth);
    EXPECT_EQ(bmb2.baseWidth + bmb3.baseWidth, bmb1.baseWidth);
    EXPECT_EQ(bmb2.numPulses + bmb3.numPulses, bmb1.numPulses);
    EXPECT_EQ(bmb2.numBases + bmb3.numBases, bmb1.numBases);
    EXPECT_FLOAT_EQ(bmb2.pkmid[3], bmb1.pkmid[3]);
    EXPECT_FLOAT_EQ(bmb2.baselineVar, bmb1.baselineVar);
    EXPECT_EQ(bmb2.numPkmidFrames[1] + bmb3.numPkmidFrames[1], bmb1.numPkmidFrames[1]);
    EXPECT_EQ(bmb2.numBaselineFrames + bmb3.numBaselineFrames, bmb1.numBaselineFrames);
}

