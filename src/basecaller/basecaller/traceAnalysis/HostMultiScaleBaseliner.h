#ifndef MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H
#define MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H

#include "MultiScaleBaseliner.h"
#include "BaselinerParams.h"
#include "BlockFilterStage.h"
#include "TraceFilters.h"

#include <common/AlignedCircularBuffer.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HostMultiScaleBaseliner : public MultiScaleBaseliner
{
    using Parent = MultiScaleBaseliner;
public:
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using FloatArray = Data::BaselinerStatAccumulator<ElementTypeOut>::FloatArray;
    using BoolArray = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;

public:     // Static functions
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::AnalysisConfig&);

    static void Finalize();

    static float SigmaEmaAlpha()
    { return sigmaEmaAlpha_; }

public:
    HostMultiScaleBaseliner(uint32_t poolId,
                            const BaselinerParams& params,
                            uint32_t lanesPerPool)
        : MultiScaleBaseliner(poolId, params, lanesPerPool)
    {
       baselinerByLane_.reserve(lanesPerPool);
       for (uint32_t l = 0; l < lanesPerPool; l++)
       {
           baselinerByLane_.emplace_back(params, Scale(), pedestal_);
       }
    }

    HostMultiScaleBaseliner(const HostMultiScaleBaseliner&) = delete;
    HostMultiScaleBaseliner(HostMultiScaleBaseliner&&) = default;
    ~HostMultiScaleBaseliner() override = default;

private:

    std::pair<Data::TraceBatch<ElementTypeOut>, Data::BaselinerMetrics>
    FilterBaseline(const Data::TraceBatchVariant& rawTrace) override;

private:

    class LaneBaseliner
    {
    public:
        LaneBaseliner(const BaselinerParams& params, float scaler, int16_t pedestal)
            : msLowerOpen_(params.Strides(), params.Widths())
            , msUpperOpen_(params.Strides(), params.Widths())
            , cMeanBias_{params.MeanBias() * std::exp2(cMeanBiasAdj_)}
            , cSigmaBias_{params.SigmaBias() * std::exp2(cSigmaBiasAdj_)}
            , stride_(params.AggregateStride())
            , scaler_(scaler)
            , pedestal_(pedestal)
        { }

        LaneBaseliner(const LaneBaseliner&) = delete;
        LaneBaseliner(LaneBaseliner&&) = default;
        ~LaneBaseliner() = default;

    public:
        size_t Stride() const { return stride_; }

        template <typename T>
        Data::BaselinerStatAccumulator<ElementTypeOut> EstimateBaseline(const Data::BlockView<const T>& traceData,
                                                                        Data::BlockView<ElementTypeOut> lowerBuffer,
                                                                        Data::BlockView<ElementTypeOut> upperBuffer,
                                                                        Data::BlockView<ElementTypeOut> baselineSubtractedData);

        void AddToBaselineStats(const LaneArray& traceData,
                                const LaneArray& baselineSubtractedFrames,
                                Data::BaselinerStatAccumulator<ElementTypeOut>& baselinerStats);

        FloatArray GetSmoothedBlEstimate(const LaneArray& lower, const LaneArray& upper);

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
                , second(widths[0], 1, strides[0])
            {
                assert(strides.size() == widths.size());
                for (size_t i = 1; i < strides.size(); ++i)
                {
                    firstv.emplace_back(BlockFilterStage<VIn, ErodeHgw<VIn>>(widths[i], strides[i], strides[i-1]));
                    secondv.emplace_back(BlockFilterStage<VIn, DilateHgw<VIn>>(widths[i], 1, strides[i-1]*strides[i]));
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
                    firstv[i](input);
                    secondv[i](input);
                }
                return input;
            }

            typename FilterTraits<VIn, filter>::FirstStage first;
            typename FilterTraits<VIn, filter>::SecondStage second;

            std::vector<BlockFilterStage<VIn, ErodeHgw<VIn>>> firstv;
            std::vector<BlockFilterStage<VIn, DilateHgw<VIn>>> secondv;
        };  // MultiStageFilter

    private:
        MultiStageFilter<ElementTypeOut, FilterType::Lower> msLowerOpen_;
        MultiStageFilter<ElementTypeOut, FilterType::Upper> msUpperOpen_;

        const FloatArray cMeanBias_;
        const FloatArray cSigmaBias_;
        // Unbiased exponential moving average used to smooth estimates of
        // baseline mean.
        FloatArray blMeanUemaSum_ = 0.0f;

        // Normalization factor for UEMA. See blMeanUemaSum_.
        FloatArray blMeanUemaWeight_ = 0.0f;

        // todo: This initial value (in DN) should be set via some set of
        // system parameters.
        FloatArray blSigmaEma_ = 2.0;

        LaneArray latData_{0};
        LaneArray latRawData_{0};
        BoolArray latLMask_{false};
        BoolArray latHMask2_{false};
        BoolArray latHMask1_{false};

        const size_t stride_;
        float scaler_;  // Converts DN quantization to e- values
        int16_t pedestal_;
    };  // LaneBaseliner

private:
    std::vector<LaneBaseliner> baselinerByLane_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H
