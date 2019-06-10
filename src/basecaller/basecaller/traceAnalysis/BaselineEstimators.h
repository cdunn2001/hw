#ifndef mongo_basecaller_traceAnalysis_BaselineEstimators_H_
#define mongo_basecaller_traceAnalysis_BaselineEstimators_H_

#include "Baseliner.h"
#include "BaselinerParams.h"
#include "BlockFilterStage.h"
#include "TraceFilters.h"

#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class NoOpBaseliner : public Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeIn = Parent::ElementTypeIn;
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using FloatArray = Data::BaselinerStatAccumulator<ElementTypeOut>::FloatArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;

public:
    NoOpBaseliner(uint32_t poolId)
        : Baseliner(poolId)
    { }

    NoOpBaseliner(const NoOpBaseliner&) = delete;
    NoOpBaseliner(NoOpBaseliner&&) = default;
    ~NoOpBaseliner() noexcept= default;

private:
    Data::CameraTraceBatch Process(Data::TraceBatch<ElementTypeIn> rawTrace) override;
};

class MultiScaleBaseliner : public Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeIn = Parent::ElementTypeIn;
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using FloatArray = Data::BaselinerStatAccumulator<ElementTypeOut>::FloatArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;

public:
    MultiScaleBaseliner(uint32_t poolId, float scaler, const BaselinerParams& config);

    MultiScaleBaseliner(const MultiScaleBaseliner&) = delete;
    MultiScaleBaseliner(MultiScaleBaseliner&&) = default;
    ~MultiScaleBaseliner() noexcept = default;

private:
    Data::CameraTraceBatch Process(Data::TraceBatch<ElementTypeIn> rawTrace) override;

private:    // Multi-stage filter
    enum class FilterType
    {
        Upper,
        Lower,
    };

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

    template <typename VIn, FilterType filter>
    struct MultiStageFilter
    {
        using InputContainer = Data::BlockView<VIn>;

        MultiStageFilter(const std::vector<size_t>& strides, const std::vector<size_t>& widths)
                : first(widths[0], strides[0])
                  , second(widths[0])
        {
            assert(strides.size() == widths.size());
            for (size_t i = 1; i < strides.size(); ++i)
            {
                firstv.emplace_back(new BlockFilterStage<VIn, ErodeHgw<VIn>>(widths[i], strides[i]));
                secondv.emplace_back(new BlockFilterStage<VIn, DilateHgw<VIn>>(widths[i]));
            }
        }

        MultiStageFilter(const MultiStageFilter&) = delete;
        MultiStageFilter& operator=(const MultiStageFilter&) = delete;
        MultiStageFilter(MultiStageFilter&&) = default;
        MultiStageFilter& operator=(MultiStageFilter&&) = default;

        InputContainer* operator()(InputContainer* input)
        {
            first(input);
            second(input);
            for (size_t i = 0; i < firstv.size(); ++i)
            {
                (*firstv[i])(input);
                (*secondv[i])(input);
            }
            return input;
        }

        typename FilterTraits<VIn, filter>::FirstStage first;
        typename FilterTraits<VIn, filter>::SecondStage second;

        // These unique_ptrs are an unfortunate extra indirection necessitated
        // because icc compiles as if it was gcc 4.7 when mpss 3.5 is installed.
        // The ptrs can probably be removed once we upgrade to a newer mpss
        std::vector<std::unique_ptr<BlockFilterStage<VIn, ErodeHgw<VIn>>>> firstv;
        std::vector<std::unique_ptr<BlockFilterStage<VIn, DilateHgw<VIn>>>> secondv;
    };

private:
    MultiStageFilter<ElementTypeIn, FilterType::Lower> msLowerOpen_;
    MultiStageFilter<ElementTypeIn, FilterType::Upper> msUpperOpen_;

    const size_t stride_;
    const FloatArray cSigmaBias_;
    const FloatArray cMeanBias_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_BaselineEstimators_H_
