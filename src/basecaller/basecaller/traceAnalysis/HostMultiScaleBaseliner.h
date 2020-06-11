#ifndef MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H
#define MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H

#include "Baseliner.h"
#include "BaselinerParams.h"
#include "BlockFilterStage.h"
#include "TraceFilters.h"

#include <common/AlignedCircularBuffer.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class HostMultiScaleBaseliner : public Baseliner
{
    using Parent = Baseliner;
public:
    using ElementTypeIn = Parent::ElementTypeIn;
    using ElementTypeOut = Parent::ElementTypeOut;
    using LaneArray = Data::BaselinerStatAccumulator<ElementTypeOut>::LaneArray;
    using FloatArray = Data::BaselinerStatAccumulator<ElementTypeOut>::FloatArray;
    using Mask = Data::BaselinerStatAccumulator<ElementTypeOut>::Mask;

public:
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig&);


    static void Finalize();

public:
    HostMultiScaleBaseliner(uint32_t poolId, float scaler, const BaselinerParams& config, uint32_t lanesPerPool)
        : Baseliner(poolId, scaler)
    {
       baselinerByLane_.reserve(lanesPerPool);
       for (uint32_t l = 0; l < lanesPerPool; l++)
       {
           baselinerByLane_.emplace_back(config, scaler);
       }
    }

    HostMultiScaleBaseliner(const HostMultiScaleBaseliner&) = delete;
    HostMultiScaleBaseliner(HostMultiScaleBaseliner&&) = default;
    ~HostMultiScaleBaseliner() override = default;

private:

    std::pair<Data::TraceBatch<ElementTypeOut>, Data::BaselinerMetrics>
    Process(const Data::TraceBatch<ElementTypeIn>& rawTrace) override;

private:

    class MultiScaleBaseliner
    {
    public:
        MultiScaleBaseliner(const BaselinerParams& config, float scaler)
            : msLowerOpen_(config.Strides(), config.Widths())
            , msUpperOpen_(config.Strides(), config.Widths())
            , stride_(config.AggregateStride())
            , cSigmaBias_{config.SigmaBias()}
            , cMeanBias_{config.MeanBias()}
            , scaler_(scaler)
        {
            latHMask_.push_back(Mask{false});
            latHMask_.push_back(Mask{false});
        }

        MultiScaleBaseliner(const MultiScaleBaseliner&) = delete;
        MultiScaleBaseliner(MultiScaleBaseliner&&) = default;
        ~MultiScaleBaseliner() = default;

    public:
        size_t Stride() const { return stride_; }

        float Scale() const { return scaler_; }

        Data::BaselinerStatAccumulator<ElementTypeOut> EstimateBaseline(const Data::BlockView<const ElementTypeIn>& traceData,
                                                                        Data::BlockView<ElementTypeIn> lowerBuffer,
                                                                        Data::BlockView<ElementTypeIn> upperBuffer,
                                                                        Data::BlockView<ElementTypeOut> baselineSubtractedData);

        void AddToBaselineStats(const LaneArray& traceData,
                                const LaneArray& baselineSubtractedFrames,
                                Data::BaselinerStatAccumulator<ElementTypeOut>& baselinerStats);

        FloatArray GetSmoothedSigma(const FloatArray& sigma);

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
        };  // MultiStageFilter

    private:
        MultiStageFilter<ElementTypeIn, FilterType::Lower> msLowerOpen_;
        MultiStageFilter<ElementTypeIn, FilterType::Upper> msUpperOpen_;

        const size_t stride_;
        const FloatArray cSigmaBias_;
        const FloatArray cMeanBias_;
        float scaler_;

        FloatArray prevSigma_;
        bool firstFrame_ = true;
        LaneArray latData_;
        LaneArray latRawData_;
        Mask latLMask_{false};
        AlignedCircularBuffer<Mask> latHMask_{2};
        FloatArray thrLow_;
        FloatArray thrHigh_;

    };  // MultiScaleBaseliner

private:
    std::vector<MultiScaleBaseliner> baselinerByLane_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H
