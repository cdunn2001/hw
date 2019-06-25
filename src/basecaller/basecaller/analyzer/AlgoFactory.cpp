
#include "AlgoFactory.h"

#include <sstream>
#include <pacbio/PBException.h>

#include <basecaller/traceAnalysis/BaselinerParams.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/DeviceSGCFrameLabeler.h>
#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/HostNoOpBaseliner.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>

#include <dataTypes/MovieConfig.h>

using std::ostringstream;
using std::unique_ptr;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

AlgoFactory::AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig)
{
    // Baseliner
    baselinerOpt_ = bcConfig.baselinerConfig.Method;


    frameLabelerOpt_ = bcConfig.frameLabelerConfig.Method;

    // Histogram accumulator
    histAccumOpt_ = bcConfig.traceHistogramConfig.Method;

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
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        DeviceSGCFrameLabeler::Finalize();
        break;
    default:
        PBLOG_ERROR << "Unrecognized method option for FrameLabeler: "
                    << frameLabelerOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
    }
}


void AlgoFactory::Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                            const Data::MovieConfig& movConfig)
{
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
    }

    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        DeviceSGCFrameLabeler::Configure(Data::GetPrimaryConfig().lanesPerPool,
                                         Data::GetPrimaryConfig().framesPerChunk);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for FrameLabeler: " << frameLabelerOpt_.toString() << '.';
        throw PBException(msg.str());
    }

    // TODO: Configure other algorithms according to options.
    TraceHistogramAccumulator::Configure(bcConfig.traceHistogramConfig, movConfig);
    DetectionModelEstimator::Configure(bcConfig.dmeConfig, movConfig);
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
            return std::unique_ptr<Baseliner>(new HostNoOpBaseliner(poolId));
        case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
            return std::make_unique<DeviceMultiScaleBaseliner>(poolId, Data::GetPrimaryConfig().lanesPerPool);
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
            // TODO: scaler currently set to default 1.0f
            return std::unique_ptr<Baseliner>(new HostMultiScaleBaseliner(poolId, 1.0f,
                                                                          FilterParamsLookup(baselinerOpt_),
                                                                          Data::GetPrimaryConfig().lanesPerPool));
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
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        return std::make_unique<DeviceSGCFrameLabeler>(poolId);
    default:
        ostringstream msg;
        msg << "Unrecognized method option for FrameLabeler: " << frameLabelerOpt_.toString() << '.';
        throw PBException(msg.str());
    }
}

unique_ptr<TraceHistogramAccumulator>
AlgoFactory::CreateTraceHistAccumulator(unsigned int poolId) const
{
    switch (histAccumOpt_)
    {
    case Data::BasecallerTraceHistogramConfig::MethodName::Host:
        return unique_ptr<TraceHistogramAccumulator>(new TraceHistogramAccumHost(poolId));
    case Data::BasecallerTraceHistogramConfig::MethodName::Gpu:
        // TODO: For now fall through to throw exception.
    default:
        ostringstream msg;
        msg << "Unrecognized method option for TraceHistogramAccumulator: " << histAccumOpt_ << '.';
        throw PBException(msg.str());
    }
}
}}}     // namespace PacBio::Mongo::Basecaller
