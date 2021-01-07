#ifndef mongo_basecaller_analyzer_AlgoFactory_H_
#define mongo_basecaller_analyzer_AlgoFactory_H_

#include <memory>

#include <common/cuda/memory/DeviceAllocationStash.h>

#include <basecaller/traceAnalysis/TraceAnalysisForward.h>
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
    AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig);

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
    Data::BasecallerFrameLabelerConfig::MethodName frameLabelerOpt_;
    Data::BasecallerTraceHistogramConfig::MethodName histAccumOpt_;
    Data::BasecallerBaselineStatsAggregatorConfig::MethodName baselineStatsAggregatorOpt_;
    Data::BasecallerDmeConfig::MethodName dmeOpt_;
    Data::BasecallerPulseAccumConfig::MethodName pulseAccumOpt_;
    Data::BasecallerMetricsConfig::MethodName hfMetricsOpt_;

    // TODO: Add enums for other strategy options as needed.
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_analyzer_AlgoFactory_H_
