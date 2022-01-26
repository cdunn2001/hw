// Copyright (c) 2019-2022, Pacific Biosciences of California, Inc.
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

#include <common/cuda/streams/LaunchManager.cuh>

#include <basecaller/traceAnalysis/FrameLabelerDeviceDataStructures.cuh>
#include <basecaller/traceAnalysis/FrameLabelerKernels.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Mongo::Data;

namespace PacBio::Mongo::Basecaller {

void FrameLabelerDevice::Configure(const Data::AnalysisConfig& analysisConfig,
                                   const Data::BasecallerFrameLabelerConfig& labelerConfig)
{
    const auto hostExecution = false;
    InitFactory(hostExecution, ViterbiStitchLookback);

    Subframe::TransitionMatrix<half> transHost(analysisConfig.movieInfo.analogs,
                                               labelerConfig.viterbi,
                                               analysisConfig.movieInfo.frameRate);
    CudaCopyToSymbol(trans, &transHost);
}

void FrameLabelerDevice::Finalize() {}

class FrameLabelerDevice::Impl
{
    static constexpr size_t BlockThreads = laneSize/2;

    static BatchDimensions LatBatchDims(size_t lanesPerPool)
    {
        BatchDimensions ret;
        ret.framesPerBatch = ViterbiStitchLookback;
        ret.laneWidth = laneSize;
        ret.lanesPerBatch = lanesPerPool;
        return ret;
    }

public:

    Impl(size_t lanesPerPool, StashableAllocRegistrar* registrar= nullptr)
        : latent_(registrar, SOURCE_MARKER(), lanesPerPool)
        , prevLat_(LatBatchDims(lanesPerPool), SyncDirection::HostReadDeviceWrite, SOURCE_MARKER())
    {
        PBLauncher(InitLatent, lanesPerPool, BlockThreads)(prevLat_);
        CudaSynchronizeDefaultStream();
    }

    void ProcessBatch(const UnifiedCudaArray<LaneModelParameters>& models,
                      const Mongo::Data::BatchData<int16_t>& input,
                      Mongo::Data::BatchData<int16_t>& latOut,
                      Mongo::Data::BatchData<int16_t>& output,
                      Mongo::Data::FrameLabelerMetrics& metricsOutput)
    {

        ViterbiDataHost<BlockThreads> labels(input.NumFrames() + ViterbiStitchLookback, input.LanesPerBatch());

        const auto& launcher = PBLauncher(FrameLabelerKernel<BlockThreads>, input.LanesPerBatch(), BlockThreads);
        launcher(models,
                 input,
                 latent_,
                 labels,
                 prevLat_,
                 latOut,
                 output,
                 metricsOutput.viterbiScore);

        CudaSynchronizeDefaultStream();
        std::swap(prevLat_, latOut);
    }

private:
    DeviceOnlyArray<LatentViterbi<BlockThreads>> latent_;
    Mongo::Data::BatchData<int16_t> prevLat_;
};

constexpr size_t FrameLabelerDevice::Impl::BlockThreads;

FrameLabelerDevice::FrameLabelerDevice(uint32_t poolId,
                                       uint32_t lanesPerPool,
                                       StashableAllocRegistrar* registrar)
    : FrameLabeler(poolId)
    , labeler_{std::make_unique<Impl>(lanesPerPool, registrar)}
{}

FrameLabelerDevice::~FrameLabelerDevice() = default;

std::pair<LabelsBatch, FrameLabelerMetrics>
FrameLabelerDevice::Process(TraceBatch<Data::BaselinedTraceElement> trace,
                            const PoolModelParameters& models)
{
    auto ret = batchFactory_->NewBatch(std::move(trace));

    labeler_->ProcessBatch(models,
                           ret.first.TraceData(),
                           ret.first.LatentTrace(),
                           ret.first,
                           ret.second);

    // Update the trace data so downstream filters can't see the held back portion
    ret.first.TraceData().SetFrameLimit(ret.first.NumFrames() - ViterbiStitchLookback);

    return ret;
}

}  // namespace PacBio::Mongo::Basecaller
