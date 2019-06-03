#ifndef mongo_basecaller_analyzer_AlgoFactory_H_
#define mongo_basecaller_analyzer_AlgoFactory_H_

#include <memory>

#include <basecaller/traceAnalysis/TraceAnalysisForward.h>
#include <dataTypes/ConfigForward.h>
#include <dataTypes/BasecallerConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class AlgoFactory
{
public:     // Static functions

public:     // Structors and assignment
    // TODO: Should constructor handling configuration?
    AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig);

    // TODO: Add a destructor that calls the static "finalize" funtions of each algorithm.

public:
    void Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                   const Data::MovieConfig& movConfig);

    std::unique_ptr<Baseliner> CreateBaseliner(unsigned int poolId) const;

    // TODO: Add Create* functions for other strategy interfaces.

private:
    Data::BasecallerBaselinerConfig::MethodName baselinerOpt_;

    // TODO: Add enums for other strategy options as needed.
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_analyzer_AlgoFactory_H_
