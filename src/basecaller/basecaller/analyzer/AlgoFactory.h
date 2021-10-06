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

#ifndef mongo_basecaller_analyzer_AlgoFactory_H_
#define mongo_basecaller_analyzer_AlgoFactory_H_

#include <memory>

#include <common/cuda/memory/DeviceAllocationStash.h>

#include <basecaller/traceAnalysis/TraceAnalysisForward.h>
#include <basecaller/traceAnalysis/TraceInputProperties.h>
#include <dataTypes/BatchData.h>
#include <dataTypes/configs/BasecallerAlgorithmConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class AlgoFactory
{
public:     // Static functions

public:     // Structors and assignment
    // TODO: Should constructor handling configuration?
    AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig,
                const TraceInputProperties& expectedTraceInfo);

    ~AlgoFactory();

public:
    void Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                   const Data::MovieConfig& movConfig);

    std::unique_ptr<Baseliner> CreateBaseliner(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;
    std::unique_ptr<FrameLabeler> CreateFrameLabeler(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;
    std::unique_ptr<PulseAccumulator> CreatePulseAccumulator(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;
    std::unique_ptr<HFMetricsFilter> CreateHFMetricsFilter(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;
    std::unique_ptr<TraceHistogramAccumulator>
    CreateTraceHistAccumulator(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;
    std::unique_ptr<BaselineStatsAggregator>
    CreateBaselineStatsAggregator(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;
    std::unique_ptr<CoreDMEstimator>
    CreateCoreDMEstimator(
            unsigned int poolId,
            const Data::BatchDimensions& dims,
            Cuda::Memory::StashableAllocRegistrar& registrar) const;

    // TODO: Add Create* functions for other strategy interfaces.

private:
    Data::BasecallerBaselinerConfig::MethodName baselinerOpt_;
    Data::BasecallerBaselinerConfig::FilterTypes baselinerType_;
    Data::BasecallerFrameLabelerConfig::MethodName frameLabelerOpt_;
    Data::BasecallerTraceHistogramConfig::MethodName histAccumOpt_;
    Data::BasecallerBaselineStatsAggregatorConfig::MethodName baselineStatsAggregatorOpt_;
    Data::BasecallerDmeConfig::MethodName dmeOpt_;
    Data::BasecallerPulseAccumConfig::MethodName pulseAccumOpt_;
    Data::BasecallerMetricsConfig::MethodName hfMetricsOpt_;

    TraceInputProperties expectedTraceInfo_;

    // TODO: Add enums for other strategy options as needed.
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_analyzer_AlgoFactory_H_
