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

#ifndef MONGO_BASECALLER_MULTISCALEBASELINER_H
#define MONGO_BASECALLER_MULTISCALEBASELINER_H

#include "Baseliner.h"
#include "BaselinerParams.h"
#include "BlockFilterStage.h"
#include "TraceFilters.h"

#include <common/AlignedCircularBuffer.h>
#include <dataTypes/BaselinerStatAccumulator.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

class MultiScaleBaseliner : public Baseliner
{
    using Parent = Baseliner;

public:     // Static functions
    static void Configure(const Data::BasecallerBaselinerConfig&,
                          const Data::AnalysisConfig&);

    static void Finalize();

    static float SigmaEmaAlpha()
    { return sigmaEmaAlpha_; }

public:
    MultiScaleBaseliner(uint32_t poolId,
                            const BaselinerParams& params,
                            uint32_t lanesPerPool)
        : Baseliner(poolId)
        , lanesPerPool_(lanesPerPool)
        , latency_(params.LatentSize())
        , framesPerStride_(params.AggregateStride())
    {
    }

    static float SigmaEmaScaleStrides()
    { 
        return -1.0f / std::log2(sigmaEmaAlpha_);
    }

    static float MeanEmaScaleStrides()
    { 
        return -1.0f / std::log2(meanEmaAlpha_);
    }

    size_t StartupLatency() const override 
    { 
        // pick the longest estimated ema transient in strides
        const float maxScaleStrides = std::max(MeanEmaScaleStrides(), SigmaEmaScaleStrides());
        // todo: consider scaling the second term here for additional
        // tunability.
        return latency_ + maxScaleStrides * framesPerStride_;
    }

    MultiScaleBaseliner(const MultiScaleBaseliner&) = delete;
    MultiScaleBaseliner(MultiScaleBaseliner&&) = default;
    ~MultiScaleBaseliner() override = default;

protected:     // Static data
    static float cSigmaBiasAdj_;
    static float cMeanBiasAdj_;
    static float meanEmaAlpha_;
    static float sigmaEmaAlpha_;
    static float jumpTolCoeff_;

protected:
    uint32_t lanesPerPool_;
    size_t latency_;
    size_t framesPerStride_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // MONGO_BASECALLER_MULTISCALEBASELINER_H
