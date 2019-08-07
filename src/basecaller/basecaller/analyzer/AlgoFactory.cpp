
#include "AlgoFactory.h"

#include <sstream>
#include <pacbio/PBException.h>

#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/BaselinerParams.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/DmeEmHost.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DevicePulseAccumulator.h>
#include <basecaller/traceAnalysis/DeviceSGCFrameLabeler.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/HFMetricsFilter.h>
#include <basecaller/traceAnalysis/HostPulseAccumulator.h>
#include <basecaller/traceAnalysis/HostSimulatedPulseAccumulator.h>
#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/HostNoOpBaseliner.h>
#include <basecaller/traceAnalysis/HostNoOpFrameLabeler.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/SubframeLabelManager.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>

#include <dataTypes/MovieConfig.h>
#include <dataTypes/PrimaryConfig.h>

using std::make_unique;
using std::ostringstream;
using std::unique_ptr;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

AlgoFactory::AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig)
{
    // Baseliner
    baselinerOpt_ = bcConfig.baselinerConfig.Method;

    // Histogram accumulator
    histAccumOpt_ = bcConfig.traceHistogramConfig.Method;

    // Detection model estimator
    dmeOpt_ = bcConfig.dmeConfig.Method;

    frameLabelerOpt_ = bcConfig.frameLabelerConfig.Method;
    pulseAccumOpt_ = bcConfig.pulseAccumConfig.Method;
    hfMetricsOpt_ = bcConfig.Metrics.Method;

    // TODO: Capture remaining options for algorithms.
}

AlgoFactory::~AlgoFactory()
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        HostNoOpBaseliner::Finalize();
        break;
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        DeviceMultiScaleBaseliner::Finalize();
        break;
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
        HostMultiScaleBaseliner::Finalize();
        break;
    default:
        PBLOG_ERROR << "Unrecognized method option for Baseliner: "
                    << baselinerOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
        break;
    }

    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::NoOp:
        HostNoOpFrameLabeler::Finalize();
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        DeviceSGCFrameLabeler::Finalize();
        break;
    default:
        PBLOG_ERROR << "Unrecognized method option for FrameLabeler: "
                    << frameLabelerOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
        break;
    }

    switch (pulseAccumOpt_)
    {
    case Data::BasecallerPulseAccumConfig::MethodName::NoOp:
        PulseAccumulator::Finalize();
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostSimulatedPulses:
        HostSimulatedPulseAccumulator::Finalize();
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostPulses:
        HostPulseAccumulator<SubframeLabelManager>::Finalize();
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::GpuPulses:
        DevicePulseAccumulator<SubframeLabelManager>::Finalize();
        break;
    default:
        ostringstream msg;
        PBLOG_ERROR << "Unrecognized method option for PulseAccumulator: "
                    << pulseAccumOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
        break;
    }

    switch (hfMetricsOpt_)
    {
    case Data::BasecallerMetricsConfig::MethodName::NoOp:
        NoHFMetricsFilter::Finalize();
        break;
    case Data::BasecallerMetricsConfig::MethodName::Host:
        HostHFMetricsFilter::Finalize();
        break;
    default:
        ostringstream msg;
        PBLOG_ERROR << "Unrecognized method option for HFMetricsFilter: "
                    << hfMetricsOpt_.toString()
                    << ".  Should be impossible to see this message, "
                    << "constructor should have thrown";
        break;
    }
}


void AlgoFactory::Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                            const Data::MovieConfig& movConfig)
{
    poolSize_ = Data::GetPrimaryConfig().lanesPerPool;
    chunkSize_ = Data::GetPrimaryConfig().framesPerChunk;

    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        HostNoOpBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        DeviceMultiScaleBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
        HostMultiScaleBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (dmeOpt_)
    {
    case Data::BasecallerDmeConfig::MethodName::Fixed:
        DetectionModelEstimator::Configure(bcConfig.dmeConfig, movConfig);
        break;
    case Data::BasecallerDmeConfig::MethodName::EmHost:
        DmeEmHost::Configure(bcConfig.dmeConfig, movConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for DetectionModelEstimator: "
            << dmeOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::NoOp:
        HostNoOpFrameLabeler::Configure(Data::GetPrimaryConfig().lanesPerPool,
                                        Data::GetPrimaryConfig().framesPerChunk);
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        DeviceSGCFrameLabeler::Configure(Data::GetPrimaryConfig().lanesPerPool,
                                         Data::GetPrimaryConfig().framesPerChunk);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for FrameLabeler: " << frameLabelerOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (pulseAccumOpt_)
    {
    case Data::BasecallerPulseAccumConfig::MethodName::NoOp:
        PulseAccumulator::Configure(bcConfig.pulseAccumConfig.maxCallsPerZmw);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostSimulatedPulses:
        HostSimulatedPulseAccumulator::Configure(bcConfig.pulseAccumConfig.maxCallsPerZmw);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostPulses:
        HostPulseAccumulator<SubframeLabelManager>::Configure(bcConfig.pulseAccumConfig.maxCallsPerZmw);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::GpuPulses:
        DevicePulseAccumulator<SubframeLabelManager>::Configure(bcConfig.pulseAccumConfig.maxCallsPerZmw);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for pulseAccumulator: " << pulseAccumOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    // TODO: Configure other algorithms according to options.
    TraceHistogramAccumulator::Configure(bcConfig.traceHistogramConfig, movConfig);
    HFMetricsFilter::Configure(bcConfig.Metrics.sandwichTolerance,
                               Data::GetPrimaryConfig().framesPerHFMetricBlock,
                               Data::GetPrimaryConfig().framesPerChunk,
                               Data::GetPrimaryConfig().sensorFrameRate,
                               Data::GetPrimaryConfig().realtimeActivityLabels,
                               Data::GetPrimaryConfig().lanesPerPool);
}


unique_ptr<Baseliner>
AlgoFactory::CreateBaseliner(unsigned int poolId) const
{
    // TODO: We are currently overloading BasecallerBaselinerConfig::MethodName
    // to represent both the baseliner method and param. When the GPU version
    // is ready to take params, then the Configure() above should store the params
    // and a new baseliner config option should be for selecting the actual baseliner.
    switch (baselinerOpt_)
    {
        case Data::BasecallerBaselinerConfig::MethodName::NoOp:
            return std::make_unique<HostNoOpBaseliner>(poolId);
            break;
        case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
            return std::make_unique<DeviceMultiScaleBaseliner>(poolId, Data::GetPrimaryConfig().lanesPerPool);
            break;
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
            // TODO: scaler currently set to default 1.0f
            return std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f, FilterParamsLookup(baselinerOpt_),
                                                             Data::GetPrimaryConfig().lanesPerPool);
            break;
        default:
            ostringstream msg;
            msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
            throw PBException(msg.str());
            break;
    }
}

std::unique_ptr<FrameLabeler>
AlgoFactory::CreateFrameLabeler(unsigned int poolId) const
{
    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::NoOp:
        return std::make_unique<HostNoOpFrameLabeler>(poolId);
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        return std::make_unique<DeviceSGCFrameLabeler>(poolId);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for FrameLabeler: " << frameLabelerOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }
}

unique_ptr<TraceHistogramAccumulator>
AlgoFactory::CreateTraceHistAccumulator(unsigned int poolId) const
{
    switch (histAccumOpt_)
    {
    case Data::BasecallerTraceHistogramConfig::MethodName::Host:
        return std::make_unique<TraceHistogramAccumHost>(poolId, poolSize_);
        break;
    case Data::BasecallerTraceHistogramConfig::MethodName::Gpu:
        // TODO: For now fall through to throw exception.
    default:
        ostringstream msg;
        msg << "Unrecognized method option for TraceHistogramAccumulator: "
            << histAccumOpt_ << '.';
        throw PBException(msg.str());
        break;
    }
}

std::unique_ptr<DetectionModelEstimator>
AlgoFactory::CreateDetectionModelEstimator(unsigned int poolId) const
{
    switch (dmeOpt_)
    {
    case Data::BasecallerDmeConfig::MethodName::Fixed:
        return make_unique<DetectionModelEstimator>(poolId, poolSize_);

    case Data::BasecallerDmeConfig::MethodName::EmHost:
        return make_unique<DmeEmHost>(poolId, poolSize_);

    default:
        ostringstream msg;
        msg << "Unrecognized method option for DetectionModelEstimator: "
            << dmeOpt_ << '.';
        throw PBException(msg.str());
        break;
    }
}

std::unique_ptr<PulseAccumulator>
AlgoFactory::CreatePulseAccumulator(unsigned int poolId) const
{
    switch (pulseAccumOpt_)
    {
    case Data::BasecallerPulseAccumConfig::MethodName::NoOp:
        return std::make_unique<PulseAccumulator>(poolId);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostSimulatedPulses:
        return std::make_unique<HostSimulatedPulseAccumulator>(poolId);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostPulses:
        return std::make_unique<HostPulseAccumulator<SubframeLabelManager>>(poolId, Data::GetPrimaryConfig().lanesPerPool);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::GpuPulses:
        return std::make_unique<DevicePulseAccumulator<SubframeLabelManager>>(poolId, Data::GetPrimaryConfig().lanesPerPool);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for pulseAccumulator: " << pulseAccumOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }
}

std::unique_ptr<HFMetricsFilter>
AlgoFactory::CreateHFMetricsFilter(unsigned int poolId) const
{
    switch (hfMetricsOpt_)
    {
    case Data::BasecallerMetricsConfig::MethodName::NoOp:
        return std::make_unique<NoHFMetricsFilter>(poolId);
        break;
    case Data::BasecallerMetricsConfig::MethodName::Host:
        return std::make_unique<HostHFMetricsFilter>(poolId);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for HFMetricsFilter: " << hfMetricsOpt_.toString() << '.';
        throw PBException(msg.str());
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
