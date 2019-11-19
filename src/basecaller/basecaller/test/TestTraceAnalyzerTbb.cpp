
// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//
//  Description:
//  Defines unit tests for class TraceAnalyzerTbb.

#include <vector>
#include <gtest/gtest.h>

#include <basecaller/analyzer/ITraceAnalyzer.h>
#include <dataTypes/BasecallerConfig.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/MovieConfig.h>

using std::vector;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TEST(TestTraceAnalyzerTbb, CheckMetadata)
{
    const vector<uint32_t> poolIds {2u, 3u, 5u, 8u, 1u};
    Cuda::Memory::DisablePerformanceMode();

    Data::BasecallerConfig bcConfig;
    bcConfig.algorithm.staticAnalysis = false;
    bcConfig.algorithm.baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium;
    bcConfig.algorithm.frameLabelerConfig.Method = Data::BasecallerFrameLabelerConfig::MethodName::NoOp;
    bcConfig.algorithm.pulseAccumConfig.Method = Data::BasecallerPulseAccumConfig::MethodName::NoOp;
    bcConfig.algorithm.Metrics.Method = Data::BasecallerMetricsConfig::MethodName::NoOp;

    Data::MovieConfig movConfig = Data::MockMovieConfig();

    auto traceAnalyzer = ITraceAnalyzer::Create(poolIds, bcConfig, movConfig);

    ASSERT_EQ(poolIds.size(), traceAnalyzer->NumZmwPools());

    const Data::BatchDimensions dims {64, 16};
    vector<Data::TraceBatch<int16_t>> chunk;
    vector<Data::BatchMetadata> bmdVec;

    for (const auto pid : poolIds)
    {
        const Data::BatchMetadata bmd(pid, 0, dims.framesPerBatch);
        bmdVec.push_back(bmd);
        chunk.emplace_back(bmd, dims, Cuda::Memory::SyncDirection::Symmetric, SOURCE_MARKER());
    }

    // The function under test.
    const auto bcBatch = (*traceAnalyzer)(std::move(chunk));

    ASSERT_EQ(bmdVec.size(), bcBatch.size());
    for (unsigned int i = 0; i < bcBatch.size(); ++i)
    {
        EXPECT_EQ(bmdVec[i], bcBatch[i]->pulses.GetMeta());
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
