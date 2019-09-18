#ifndef mongo_basecaller_traceAnalysis_HostNoOpBaseliner_H_
#define mongo_basecaller_traceAnalysis_HostNoOpBaseliner_H_

#include "Baseliner.h"

#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HostNoOpBaseliner : public Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeIn = Parent::ElementTypeIn;
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using FloatArray = Data::BaselinerStatAccumulator<ElementTypeOut>::FloatArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;

public:
    static void Configure(const Data::BasecallerBaselinerConfig& baselinerConfig,
                          const Data::MovieConfig& movConfig);


    static void Finalize();

public:
    HostNoOpBaseliner(uint32_t poolId)
        : Baseliner(poolId)
    { }

    HostNoOpBaseliner(const HostNoOpBaseliner&) = delete;
    HostNoOpBaseliner(HostNoOpBaseliner&&) = default;
    ~HostNoOpBaseliner() override;

private:
    std::pair<Data::TraceBatch<ElementTypeOut>, Data::BaselinerMetrics>
    Process(Data::TraceBatch<ElementTypeIn> rawTrace) override;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_HostHostNoOpBaseliner_H_
