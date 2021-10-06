#ifndef mongo_basecaller_traceAnalysis_HostNoOpBaseliner_H_
#define mongo_basecaller_traceAnalysis_HostNoOpBaseliner_H_

#include "Baseliner.h"

#include <dataTypes/BaselinerStatAccumulator.h>
#include <basecaller/traceAnalysis/TraceInputProperties.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HostNoOpBaseliner : public Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using FloatArray = Data::BaselinerStatAccumulator<ElementTypeOut>::FloatArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;

public:
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig&);


    static void Finalize();

public:
    HostNoOpBaseliner(uint32_t poolId, const TraceInputProperties& traceInfo)
        : Baseliner(poolId)
        , pedestal_(traceInfo.pedestal)
    { }

    HostNoOpBaseliner(const HostNoOpBaseliner&) = delete;
    HostNoOpBaseliner(HostNoOpBaseliner&&) = default;
    ~HostNoOpBaseliner() override;

    size_t StartupLatency() const override { return 0; }

private:
    std::pair<Data::TraceBatch<ElementTypeOut>, Data::BaselinerMetrics>
    FilterBaseline(const Data::TraceBatchVariant& rawTrace) override;

    int16_t pedestal_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_HostHostNoOpBaseliner_H_
