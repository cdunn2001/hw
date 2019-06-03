#ifndef mongo_basecaller_traceAnalysis_BaselineEstimators_H_
#define mongo_basecaller_traceAnalysis_BaselineEstimators_H_

#include "Baseliner.h"
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class NoOpBaseliner : Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeIn = Parent::ElementTypeIn;
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;
public:
    NoOpBaseliner(uint32_t poolId)
        : Baseliner(poolId)
    { }

private:
    Data::CameraTraceBatch Process(Data::TraceBatch<ElementTypeIn> rawTrace) override;
};


}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_BaselineEstimators_H_
