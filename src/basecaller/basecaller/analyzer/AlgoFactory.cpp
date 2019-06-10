
#include "AlgoFactory.h"

#include <sstream>
#include <pacbio/PBException.h>
#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DeviceSGCFrameLabeler.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <dataTypes/MovieConfig.h>

using std::ostringstream;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

AlgoFactory::AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig)
{
    // Baseliner
    baselinerOpt_ = bcConfig.baselinerConfig.Method;
    frameLabelerOpt_ = bcConfig.frameLabelerConfig.Method;

    // TODO: Capture remaining options for algorithms.
}

AlgoFactory::~AlgoFactory()
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        Baseliner::Finalize();
        break;
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        DeviceMultiScaleBaseliner::Finalize();
        break;
    default:
        ostringstream msg;
        PBLOG_ERROR << "Unrecognized method option for Baseliner: "
                    << baselinerOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
    }

    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::DeviceSubFrameGaussCaps:
        DeviceSGCFrameLabeler::Finalize();
        break;
    default:
        ostringstream msg;
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
        Baseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        DeviceMultiScaleBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
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


std::unique_ptr<Baseliner>
AlgoFactory::CreateBaseliner(unsigned int poolId) const
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        return std::make_unique<Baseliner>(poolId);
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        return std::make_unique<DeviceMultiScaleBaseliner>(poolId, Data::GetPrimaryConfig().lanesPerPool);
    default:
        ostringstream msg;
        msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
        throw PBException(msg.str());
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

}}}     // namespace PacBio::Mongo::Basecaller
