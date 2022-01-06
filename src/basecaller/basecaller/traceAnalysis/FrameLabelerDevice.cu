// Copyright (c) 2019, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "FrameLabelerDevice.h"

#include <prototypes/FrameLabeler/FrameLabelerKernels.cuh>
#include <dataTypes/configs/AnalysisConfig.h>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// static
void FrameLabelerDevice::Configure(const Data::AnalysisConfig& analysisConfig,
                                   const Data::BasecallerFrameLabelerConfig& labelerConfig)
{
    const auto hostExecution = false;
    InitFactory(hostExecution, ViterbiStitchLookback);

    Cuda::FrameLabeler::Configure(analysisConfig.movieInfo.analogs,
                                  labelerConfig,
                                  analysisConfig.movieInfo.frameRate);
}

void FrameLabelerDevice::Finalize()
{
    Cuda::FrameLabeler::Finalize();
}

FrameLabelerDevice::FrameLabelerDevice(uint32_t poolId,
                                       uint32_t lanesPerPool,
                                       StashableAllocRegistrar* registrar)
    : FrameLabeler(poolId)
    , labeler_(std::make_unique<Cuda::FrameLabeler>(lanesPerPool, registrar))
{}

FrameLabelerDevice::~FrameLabelerDevice() = default;

std::pair<LabelsBatch, FrameLabelerMetrics>
FrameLabelerDevice::Process(TraceBatch<Data::BaselinedTraceElement> trace,
                            const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));
    labeler_->ProcessBatch(
            models, ret.first.TraceData(), ret.first.LatentTrace(), ret.first, ret.second);

    // Update the trace data so downstream filters can't see the held back portion
    ret.first.TraceData().SetFrameLimit(ret.first.NumFrames() - ViterbiStitchLookback);

    return ret;
}

}}}     // namespace PacBio::Mongo::Basecaller
