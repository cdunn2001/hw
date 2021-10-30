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

#include "PulseAccumulator.h"

#include <dataTypes/configs/BasecallerPulseAccumConfig.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

std::unique_ptr<Data::PulseBatchFactory> PulseAccumulator::batchFactory_;
float PulseAccumulator::widthThresh_;
float PulseAccumulator::ampThresh_;

// static

void PulseAccumulator::Configure(const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    InitFactory(true, pulseConfig);
}

void PulseAccumulator::Finalize() {}

void PulseAccumulator::InitFactory(bool hostExecution,
                                   const Data::BasecallerPulseAccumConfig& pulseConfig)
{
    using Cuda::Memory::SyncDirection;
    SyncDirection syncDir = hostExecution ? SyncDirection::HostWriteDeviceRead : SyncDirection::HostReadDeviceWrite;
    batchFactory_ = std::make_unique<Data::PulseBatchFactory>(pulseConfig.maxCallsPerZmw, syncDir);

    const float w = pulseConfig.XspWidthThresh;
    if (w < 0.0f)
    {
        std::stringstream msg;
        msg << "Bad XspWidthThresh: " << w << ". Must be non-negative.";
        throw PBException(msg.str());
    }

    widthThresh_ = w;
    // PBLOG_INFO << "XspWidthThresh: " << w << '.';

    const double x = pulseConfig.XspAmpThresh;
    if (x < 0.0 || x > 1.0)
    {
        std::stringstream msg;
        msg << "Bad XspAmpThresh: " << x << ".  Valid range is [0, 1].";
        throw PBException(msg.str());
    }

    ampThresh_ = static_cast<float>(x / (1.0 - x));
    // PBLOG_INFO << "XspAmpThresh: " << x << ".";
}

std::pair<Data::PulseBatch, Data::PulseDetectorMetrics>
PulseAccumulator::Process(Data::LabelsBatch labels, const PoolModelParameters&)
{
    auto ret = batchFactory_->NewBatch(labels.Metadata(), labels.StorageDims());

    for (size_t laneIdx = 0; laneIdx < labels.LanesPerBatch(); ++laneIdx)
    {
        auto lanePulses = ret.first.Pulses().LaneView(laneIdx);
        lanePulses.Reset();
    }

    // Need to make sure any potential kernels populating `labels`
    // finish before we destroy the object.
    // TODO: Would be nice if we could rely on the ~LabelsBatch to handle this.
    assert(labels.LanesPerBatch() > 0);
    labels.GetBlockView(0);

    return ret;
}

PulseAccumulator::PulseAccumulator(uint32_t poolId)
    : poolId_ (poolId)
{

}

}}}     // namespace PacBio::Mongo::Basecaller
