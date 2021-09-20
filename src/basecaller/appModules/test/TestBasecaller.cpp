
// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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
//  Defines unit tests for class Basecaller.

#include <vector>
#include <gtest/gtest.h>

#include <pacbio/logging/Logger.h>

#include <appModules/Basecaller.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>

using std::vector;

using namespace PacBio;
using namespace PacBio::Mongo;

TEST(TestBasecaller, CheckMetadata)
{
    Logging::LogSeverityContext logContext (Logging::LogLevel::WARN);
    Cuda::Memory::SetGlobalAllocationMode(Cuda::Memory::CachingMode::DISABLED,
                                          Cuda::Memory::AllocatorMode::MALLOC);

    Data::SmrtBasecallerConfig config{};
    auto& algoConfig = config.algorithm;
    const Data::BatchDimensions bDims {8, 16};
    const vector<uint32_t> poolIds = {2u, 3u, 5u, 8u, 1u};
    const auto& dimMap = [&](){
        std::map<uint32_t, Data::BatchDimensions> ret;
        for (auto id : poolIds) ret[id] = bDims;
        return ret;
    }();

    algoConfig.modelEstimationMode = Data::BasecallerAlgorithmConfig::ModelEstimationMode::InitialEstimations;
    algoConfig.baselinerConfig.Method = Data::BasecallerBaselinerConfig::MethodName::HostMultiScale;
    algoConfig.baselinerConfig.Filter = Data::BasecallerBaselinerConfig::FilterTypes::TwoScaleMedium;
    algoConfig.frameLabelerConfig.Method = Data::BasecallerFrameLabelerConfig::MethodName::NoOp;
    algoConfig.pulseAccumConfig.Method = Data::BasecallerPulseAccumConfig::MethodName::NoOp;
    algoConfig.Metrics.Method = Data::BasecallerMetricsConfig::MethodName::NoOp;

    Data::MovieConfig movConfig = Data::MockMovieConfig();

    auto basecaller = Application::BasecallerBody(dimMap, algoConfig, movConfig, config.system);

    vector<Data::TraceVariant> chunk;
    vector<Data::BatchMetadata> bmdVec;
    for (const auto pid : poolIds)
    {
        const Data::BatchMetadata bmd(pid, 0, bDims.framesPerBatch, pid*bDims.ZmwsPerBatch());
        bmdVec.push_back(bmd);
        Data::TraceBatch<int16_t> batch(bmd,
                                        bDims,
                                        Cuda::Memory::SyncDirection::Symmetric,
                                        SOURCE_MARKER());
        chunk.push_back(std::move(batch));
    }

    for (unsigned int i = 0; i < chunk.size(); ++i)
    {
        auto result = basecaller.Process(chunk[i]);
        EXPECT_EQ(bmdVec[i], result.pulses.GetMeta());
    }
}
