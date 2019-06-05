
#include "AlgoFactory.h"

#include <sstream>
#include <pacbio/PBException.h>
#include <basecaller/traceAnalysis/Baseliner.h>
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

AlgoFactory::~AlgoFactory()
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        Baseliner::Finalize();
        break;
    default:
        ostringstream msg;
        PBLOG_ERROR << "Unrecognized method option for Baseliner: "
                    << baselinerOpt_.toString()
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
        return std::unique_ptr<Baseliner>(new Baseliner(poolId));
    default:
        ostringstream msg;
        msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
        throw PBException(msg.str());
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
