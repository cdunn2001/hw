// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#include "PrelimHQFilter.h"

#include <bazio/MetricBlock.h>
#include <bazio/encoding/PulseToBaz.h>
#include <bazio/writing/BazBuffer.hpp>
#include <pacbio/smrtdata/Basecall.h>

#include <dataTypes/configs/PrelimHQConfig.h>
#include <dataTypes/Pulse.h>

using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo;
using namespace PacBio::BazIO;

namespace PacBio {
namespace Application {

namespace {

void ConvertMetric(const std::unique_ptr<BatchResult::MetricsT>& metricsPtr,
                   Primary::SpiderMetricBlock& sm,
                   size_t laneIndex,
                   size_t zmwIndex)
{
    if (metricsPtr)
    {
        const auto& metrics = metricsPtr->GetHostView()[laneIndex];

        sm.ActivityLabel(static_cast<Primary::ActivityLabeler::Activity>(metrics.activityLabel[zmwIndex]));
        sm.TraceAutocorr(metrics.autocorrelation[zmwIndex]);

        sm.BpzvarA(metrics.bpZvar[0][zmwIndex])
            .BpzvarC(metrics.bpZvar[1][zmwIndex])
            .BpzvarG(metrics.bpZvar[2][zmwIndex])
            .BpzvarT(metrics.bpZvar[3][zmwIndex]);

        sm.PkzvarA(metrics.pkZvar[0][zmwIndex])
            .PkzvarC(metrics.pkZvar[1][zmwIndex])
            .PkzvarG(metrics.pkZvar[2][zmwIndex])
            .PkzvarT(metrics.pkZvar[3][zmwIndex]);

        sm.BaseWidth(metrics.numBaseFrames[zmwIndex]);
        sm.PulseWidth(metrics.numPulseFrames[zmwIndex]);

        sm.NumBasesA(metrics.numBasesByAnalog[0][zmwIndex])
            .NumBasesC(metrics.numBasesByAnalog[1][zmwIndex])
            .NumBasesG(metrics.numBasesByAnalog[2][zmwIndex])
            .NumBasesT(metrics.numBasesByAnalog[3][zmwIndex]);

        sm.NumPulses(metrics.numPulses[zmwIndex]);

        sm.NumPkmidBasesA(metrics.numPkMidBasesByAnalog[0][zmwIndex])
            .NumBasesC(metrics.numPkMidBasesByAnalog[1][zmwIndex])
            .NumBasesG(metrics.numPkMidBasesByAnalog[2][zmwIndex])
            .NumBasesT(metrics.numPkMidBasesByAnalog[3][zmwIndex]);

        sm.NumFrames(metrics.numFrames[zmwIndex]);

        sm.NumPkmidFramesA(metrics.numPkMidFrames[0][zmwIndex])
            .NumPkmidFramesC(metrics.numPkMidFrames[1][zmwIndex])
            .NumPkmidFramesG(metrics.numPkMidFrames[2][zmwIndex])
            .NumPkmidFramesT(metrics.numPkMidFrames[3][zmwIndex]);

        sm.NumPulseLabelStutters(metrics.numPulseLabelStutters[zmwIndex]);
        sm.NumHalfSandwiches(metrics.numHalfSandwiches[zmwIndex]);
        sm.NumSandwiches(metrics.numSandwiches[zmwIndex]);
        sm.PulseDetectionScore(metrics.pulseDetectionScore[zmwIndex]);

        sm.PkmaxA(metrics.pkMax[0][zmwIndex])
            .PkmaxC(metrics.pkMax[1][zmwIndex])
            .PkmaxG(metrics.pkMax[2][zmwIndex])
            .PkmaxT(metrics.pkMax[3][zmwIndex]);

        sm.PixelChecksum(metrics.pixelChecksum[zmwIndex]);;

        sm.PkmidA(metrics.pkMidSignal[0][zmwIndex])
            .PkmidC(metrics.pkMidSignal[1][zmwIndex])
            .PkmidG(metrics.pkMidSignal[2][zmwIndex])
            .PkmidT(metrics.pkMidSignal[3][zmwIndex]);

        sm.NumBaselineFrames({metrics.numFramesBaseline[zmwIndex]});
        sm.Baselines({metrics.frameBaselineDWS[zmwIndex]});
        sm.BaselineSds({metrics.frameBaselineVarianceDWS[zmwIndex]});
    }
}

using ProductionPulses =
    PulseToBaz<Field<PacketFieldName::Label,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<TruncateOverflow, NumBits_t<2>>
                     >,
               Field<PacketFieldName::Pw,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >,
               Field<PacketFieldName::StartFrame,
                     StoreSigned_t<false>,
                     Transform<DeltaCompression>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >
               >;

using InternalPulses =
    PulseToBaz<Field<PacketFieldName::Label,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<TruncateOverflow, NumBits_t<2>>
                     >,
               Field<PacketFieldName::Pw,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >,
               Field<PacketFieldName::StartFrame,
                     StoreSigned_t<false>,
                     Transform<DeltaCompression>,
                     Serialize<CompactOverflow, NumBits_t<7>>
                     >,
               Field<PacketFieldName::Pkmax,
                     StoreSigned_t<true>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<8>, NumBytes_t<2>>
                     >,
               Field<PacketFieldName::Pkmid,
                     StoreSigned_t<true>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<8>, NumBytes_t<2>>
                     >,
               Field<PacketFieldName::Pkmean,
                     StoreSigned_t<true>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<8>, NumBytes_t<2>>
               >,
               Field<PacketFieldName::Pkvar,
                     StoreSigned_t<false>,
                     Transform<FixedPoint, FixedPointScale_t<10>>,
                     Serialize<SimpleOverflow, NumBits_t<7>, NumBytes_t<2>>
               >,
               Field<PacketFieldName::IsBase,
                     StoreSigned_t<false>,
                     Transform<NoOp>,
                     Serialize<TruncateOverflow, NumBits_t<1>>
                     >
               >;

} // anon

using namespace Cuda::Memory;
using namespace PacBio::BazIO;
using namespace PacBio::Mongo::Data;

struct PrelimHQFilterBody::Impl
{
    // TODO make configurable somehow.
    static constexpr size_t expectedPulses = 164;
    Impl(size_t numZmws, size_t numBatches, const PrelimHQConfig& config)
        : numBatches_(numBatches)
        , numZmw_(numZmws)
        , chunksPerOutput_(config.bazBufferChunks)
        , zmwStride_(config.zmwOutputStride)
        , buffer_(std::make_unique<BazIO::BazBuffer>(numZmws, expectedPulses,
                                                     CreateAllocator(AllocatorMode::MALLOC, SOURCE_MARKER())))
    {
    }

    virtual std::unique_ptr<BazBuffer> Process(BatchResult in) = 0;

    size_t currFrame_ = 0;
    size_t batchesSeen_ = 0;
    size_t chunksSeen_ = 0;
    size_t numBatches_;
    size_t numZmw_;
    size_t chunksPerOutput_;
    size_t zmwStride_;
    std::unique_ptr<BazIO::BazBuffer> buffer_;
};

constexpr size_t PrelimHQFilterBody::Impl::expectedPulses;

template <bool internal>
struct PrelimHQFilterBody::ImplChild : public PrelimHQFilterBody::Impl
{
    ImplChild(size_t numZmws, size_t numBatches, const PrelimHQConfig& config)
        : Impl(numZmws, numBatches, config)
        , serializers_(numZmws)
    {}

    std::unique_ptr<BazBuffer> Process(BatchResult in) override
    {
        const auto& pulseBatch = in.pulses;
        const auto& metricsPtr = in.metrics;

        if (pulseBatch.GetMeta().FirstFrame() > currFrame_)
        {
            if (batchesSeen_ != 0)
                throw PBException("Data out of order, new chunk seen before all batches of previous chunk");
            currFrame_ = pulseBatch.GetMeta().FirstFrame();
            chunksSeen_++;
            if (chunksSeen_ == chunksPerOutput_) chunksSeen_ = 0;
        } else if (pulseBatch.GetMeta().FirstFrame() < currFrame_)
        {
            throw PBException("Data out of order, multiple chunks being processed simultaneously");
        }

        size_t currentZmwIndex = pulseBatch.GetMeta().FirstZmw();
        for (uint32_t lane = 0; lane < pulseBatch.Dims().lanesPerBatch; ++lane)
        {
            const auto& lanePulses = pulseBatch.Pulses().LaneView(lane);

            for (uint32_t zmw = 0; zmw < laneSize; zmw += zmwStride_)
            {
                if (metricsPtr)
                {
                    buffer_->AddMetrics(currentZmwIndex,
                                        [&](BazIO::MemoryBufferView<Primary::SpiderMetricBlock>& dest)
                                        {
                                            assert(dest.size() == 1);
                                            ConvertMetric(metricsPtr, dest[0], lane, zmw);
                                        },
                                        1);
                }
                auto pulses = lanePulses.ZmwData(zmw);
                buffer_->AddZmw(currentZmwIndex, pulses, pulses + lanePulses.size(zmw), serializers_[currentZmwIndex]);
                currentZmwIndex += zmwStride_;
            }
        }
        batchesSeen_++;
        std::unique_ptr<BazBuffer> ret;
        if (batchesSeen_ == numBatches_)
        {
            batchesSeen_ = 0;
            if (chunksSeen_ == chunksPerOutput_-1)
            {
                ret = std::make_unique<BazBuffer>(numZmw_, expectedPulses,
                                                  CreateAllocator(AllocatorMode::MALLOC, SOURCE_MARKER()));
                std::swap(ret, buffer_);
            }
        }
        return ret;
    }

    using Serializer = std::conditional_t<internal, InternalPulses, ProductionPulses>;
    std::vector<Serializer> serializers_;
};

PrelimHQFilterBody::PrelimHQFilterBody(size_t numZmws, size_t numBatches,
                                       const PrelimHQConfig& config, bool internal)
{

    if (internal)
        impl_ = std::make_unique<ImplChild<true>>(numZmws, numBatches, config);
    else
        impl_ = std::make_unique<ImplChild<false>>(numZmws, numBatches, config);
}

PrelimHQFilterBody::~PrelimHQFilterBody() = default;

void PrelimHQFilterBody::Process(Mongo::Data::BatchResult in)
{
    auto ret = impl_->Process(std::move(in));
    if (ret) this->PushOut(std::move(ret));
}

}}
