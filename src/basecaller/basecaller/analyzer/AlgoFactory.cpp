
#include "AlgoFactory.h"

#include <sstream>
#include <pacbio/PBException.h>
#include <basecaller/traceAnalysis/BaselineEstimators.h>
#include <basecaller/traceAnalysis/BaselinerParams.h>
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

    // TODO: Capture remaining options for algorithms.
}


void AlgoFactory::Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                            const Data::MovieConfig& movConfig)
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        Baseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
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
            return std::unique_ptr<Baseliner>(new NoOpBaseliner(poolId));
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
            // TODO: scaler currently set to default 1.0f
            return std::unique_ptr<Baseliner>(new MultiScaleBaseliner(poolId, 1.0f, FilterParamsLookup(baselinerOpt_)));
        default:
            ostringstream msg;
            msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
            throw PBException(msg.str());
            break;
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
