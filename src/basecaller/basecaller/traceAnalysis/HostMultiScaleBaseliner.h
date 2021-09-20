//  Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
//  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
//  THIS LICENSE.  THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR
//  ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
//  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.

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

public:     // Static functions
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::MovieConfig&);


    static void Finalize();

public:
    HostMultiScaleBaseliner(uint32_t poolId, float scaler, const BaselinerParams& params, uint32_t lanesPerPool)
        : Baseliner(poolId, scaler)
        , latency_(params.LatentSize())
    {
       baselinerByLane_.reserve(lanesPerPool);
       for (uint32_t l = 0; l < lanesPerPool; l++)
       {
           baselinerByLane_.emplace_back(params, scaler);
       }
    }

    size_t StartupLatency() const override { return latency_; }

    HostMultiScaleBaseliner(const HostMultiScaleBaseliner&) = delete;
    HostMultiScaleBaseliner(HostMultiScaleBaseliner&&) = default;
    ~HostMultiScaleBaseliner() override = default;

private:

    std::pair<Data::TraceBatch<ElementTypeOut>, Data::BaselinerMetrics>
    FilterBaseline_(const Data::TraceBatch<ElementTypeIn>& rawTrace) override;

private:

    class MultiScaleBaseliner
    {
    public:
        MultiScaleBaseliner(const BaselinerParams& params, float scaler)
            : msLowerOpen_(params.Strides(), params.Widths())
            , msUpperOpen_(params.Strides(), params.Widths())
            , stride_(params.AggregateStride())
            , cSigmaBias_{params.SigmaBias()}
            , cMeanBias_{params.MeanBias()}
            , scaler_(scaler)
        { }

        MultiScaleBaseliner(const MultiScaleBaseliner&) = delete;
        MultiScaleBaseliner(MultiScaleBaseliner&&) = default;
        ~MultiScaleBaseliner() = default;

    public:
        size_t Stride() const { return stride_; }

        float Scale() const { return scaler_; }    // Converts DN quantization to e- values

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
        MultiStageFilter<ElementTypeIn, FilterType::Lower> msLowerOpen_;
        MultiStageFilter<ElementTypeIn, FilterType::Upper> msUpperOpen_;

        const size_t stride_;
        const FloatArray cSigmaBias_;
        const FloatArray cMeanBias_;
        float scaler_;

        FloatArray bgSigma_{0};
        LaneArray latData_{0};
        LaneArray latRawData_{0};
        Mask latLMask_{false};
        Mask latHMask2_{false};
        Mask latHMask1_{false};
        FloatArray thrLow_{0};
        FloatArray thrHigh_{0};

    };  // MultiScaleBaseliner

private:
    std::vector<MultiScaleBaseliner> baselinerByLane_;
    size_t latency_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // MONGO_BASECALLER_HOSTMULTISCALEBASELINER_H
