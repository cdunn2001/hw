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

#include "DeviceMultiScaleBaseliner.h"
#include <dataTypes/configs/BasecallerBaselinerConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>

#include <prototypes/BaselineFilter/BaselineFilterKernels.cuh>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

using namespace PacBio::Cuda::Memory;

constexpr short  DeviceMultiScaleBaseliner::initVal;
float DeviceMultiScaleBaseliner::cMeanBiasAdj_  = 0.0f;
float DeviceMultiScaleBaseliner::cSigmaBiasAdj_ = 0.0f;
float DeviceMultiScaleBaseliner::meanEmaAlpha_  = 0.0f;
float DeviceMultiScaleBaseliner::sigmaEmaAlpha_ = 0.0f;
float DeviceMultiScaleBaseliner::jumpTolCoeff_  = std::numeric_limits<float>::infinity();

void DeviceMultiScaleBaseliner::Configure(const Data::BasecallerBaselinerConfig& bbc,
                                          const Data::AnalysisConfig& analysisConfig)
{
    const auto hostExecution = false;
    InitFactory(hostExecution, analysisConfig);

    cMeanBiasAdj_ = bbc.MeanBiasAdjust;

    cSigmaBiasAdj_ = bbc.SigmaBiasAdjust;

    const float meanEmaScale = bbc.MeanEmaScaleStrides;
    meanEmaAlpha_ = std::pow(0.5f, 1.0f / meanEmaScale);
    assert(0.0f <= meanEmaAlpha_ && meanEmaAlpha_ < 1.0f);

    const float sigmaEmaScale = bbc.SigmaEmaScaleStrides;
    std::ostringstream msg;
    msg << "SigmaEmaScaleStrides = " << sigmaEmaScale << '.';
    // TODO: Use a scoped logger.
    PBLOG_INFO << msg.str();
    sigmaEmaAlpha_ = std::exp2(-1.0f / sigmaEmaScale);
    assert(0.0f <= sigmaEmaAlpha_ && sigmaEmaAlpha_ <= 1.0f);

    // TODO: Enable jumpTolCoeff_
    // const float js = bbc.JumpSuppression;
    // jumpTolCoeff_ = (js > 0.0f ? 1.0f / js : std::numeric_limits<float>::infinity());
}

void DeviceMultiScaleBaseliner::Finalize() {}

std::pair<Data::TraceBatch<Data::BaselinedTraceElement>,
          Data::BaselinerMetrics>
DeviceMultiScaleBaseliner::FilterBaseline(const Data::TraceBatchVariant& rawTrace)
{
    auto out = batchFactory_->NewBatch(rawTrace.Metadata(), rawTrace.StorageDims());

    filter_->RunBaselineFilter(rawTrace, out.first, out.second.baselinerStats);

    const auto& tracemd = out.first.Metadata();
    out.second.frameInterval = {tracemd.FirstFrame(), tracemd.LastFrame()};

    return out;
}

DeviceMultiScaleBaseliner::DeviceMultiScaleBaseliner(uint32_t poolId,
                                                     const BaselinerParams& params,
                                                     uint32_t lanesPerPool,
                                                     StashableAllocRegistrar* registrar)
    : Baseliner(poolId)
    , latency_(params.LatentSize())
    , framesPerStride_(params.AggregateStride())
{
    Cuda::ComposedConstructArgs args;
    args.pedestal = pedestal_;
    args.scale = Scale();
    args.meanBiasAdj   = cMeanBiasAdj_;
    args.sigmaBiasAdj  = cSigmaBiasAdj_;
    args.meanEmaAlpha  = meanEmaAlpha_;
    args.sigmaEmaAlpha = sigmaEmaAlpha_;
    args.numLanes = lanesPerPool;
    args.val = initVal;
    switch (expectedEncoding_)
    {
    case DataSource::PacketLayout::EncodingFormat::UINT8:
        filter_ = std::make_unique<Cuda::ComposedFilter<laneSize/2, lag, uint8_t>>(
            params,
            args,
            SOURCE_MARKER(),
            registrar);
        break;
    case DataSource::PacketLayout::EncodingFormat::INT16:
        filter_ = std::make_unique<Cuda::ComposedFilter<laneSize/2, lag, int16_t>>(
            params,
            args,
            SOURCE_MARKER(),
            registrar);
        break;
    default:
        throw PBException("Unexpected data encoding in device baseline filter");
    }
}

DeviceMultiScaleBaseliner::~DeviceMultiScaleBaseliner() = default;

}}}      // namespace PacBio::Mongo::Basecaller
