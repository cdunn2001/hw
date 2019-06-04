#ifndef mongo_basecaller_traceAnalysis_BaselineEstimators_H_
#define mongo_basecaller_traceAnalysis_BaselineEstimators_H_

#include "Baseliner.h"
#include "BlockFilterStage.h"
#include "TraceFilters.h"

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

enum class FilterType
{
    Upper,
    Lower,
    Hybrid
};

// TODO: Finish below

/*
template <typename VIn, FilterType>
struct FilterTraits;
template <typename VIn>
struct FilterTraits<VIn, FilterType::Lower>
{
    using FirstStage  = BlockFilterStage<VIn, ErodeHgw<VIn>>;
    using SecondStage = BlockFilterStage<VIn, DilateHgw<VIn>>;
};
template <typename VIn>
struct FilterTraits<VIn, FilterType::Upper>
{
    using FirstStage  = BlockFilterStage<VIn, DilateHgw<VIn>>;
    using SecondStage = BlockFilterStage<VIn, ErodeHgw<VIn>>;
};
*/

template <typename VIn, FilterType filter>
struct MultiStageFilter
{

};

class MultiScaleBaseliner : Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeIn = Parent::ElementTypeIn;
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;
public:


private:
    Data::CameraTraceBatch Process(Data::TraceBatch<ElementTypeIn> rawTrace) override;

};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_BaselineEstimators_H_
