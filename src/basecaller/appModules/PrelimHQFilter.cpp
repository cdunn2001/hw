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
#include <dataTypes/PulseGroups.h>

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

} // anon

using namespace Cuda::Memory;
using namespace PacBio::BazIO;
using namespace PacBio::Mongo::Data;

struct PrelimHQFilterBody::Impl
{
    // TODO make configurable somehow.
    static constexpr size_t expectedPulses = 164;
    Impl(size_t numZmws, const std::map<uint32_t, BatchDimensions>& poolDims,
         const PrelimHQConfig& config)
        : numBatches_(poolDims.size())
        , numZmw_(numZmws)
        , chunksPerOutput_(config.bazBufferChunks)
        , zmwStride_(config.zmwOutputStride)
        , poolDims_(poolDims)
    { }

    virtual std::unique_ptr<BazBuffer> Process(BatchResult in) = 0;

    size_t currFrame_ = 0;
    size_t batchesSeen_ = 0;
    size_t chunksSeen_ = 0;
    size_t numBatches_;
    size_t numZmw_;
    size_t chunksPerOutput_;
    size_t zmwStride_;
    std::map<uint32_t, BatchDimensions> poolDims_;
};

constexpr size_t PrelimHQFilterBody::Impl::expectedPulses;

struct PrelimHQFilterBody::SingleBuffer
{
    SingleBuffer(PrelimHQFilterBody::Impl* p)
    : buffer_(std::make_unique<BazIO::BazBuffer>(p->numZmw_, 0,
                                                 PrelimHQFilterBody::Impl::expectedPulses,
                                                 CreateAllocator(AllocatorMode::MALLOC, SOURCE_MARKER())))
    { }

    template <typename Serializer>
    std::unique_ptr<BazBuffer> Process(PrelimHQFilterBody::Impl* p, std::vector<Serializer>& serializers, BatchResult in)
    {
        const auto& pulseBatch = in.pulses;
        const auto& metricsPtr = in.metrics;

        if (pulseBatch.GetMeta().FirstFrame() > p->currFrame_)
        {
            if (p->batchesSeen_ != 0)
                throw PBException("Data out of order, new chunk seen before all batches of previous chunk");
            p->currFrame_ = pulseBatch.GetMeta().FirstFrame();
            p->chunksSeen_++;
            if (p->chunksSeen_ == p->chunksPerOutput_) p->chunksSeen_ = 0;
        }
        else if (pulseBatch.GetMeta().FirstFrame() < p->currFrame_)
        {
            throw PBException("Data out of order, multiple chunks being processed simultaneously");
        }

        size_t currentZmwIndex = pulseBatch.GetMeta().FirstZmw();
        for (uint32_t lane = 0; lane < pulseBatch.Dims().lanesPerBatch; ++lane)
        {
            const auto& lanePulses = pulseBatch.Pulses().LaneView(lane);

            for (uint32_t zmw = 0; zmw < laneSize; zmw += p->zmwStride_)
            {
                if (metricsPtr)
                {

                    buffer_->AddMetrics(currentZmwIndex,
                                        [&](BazIO::MemoryBufferView<Primary::SpiderMetricBlock>& dest) {
                                            assert(dest.size() == 1);
                                            ConvertMetric(metricsPtr, dest[0], lane, zmw);
                                        },
                                        1);

                }
                auto pulses = lanePulses.ZmwData(zmw);
                buffer_->AddZmw(currentZmwIndex, pulses, pulses + lanePulses.size(zmw),
                                serializers[currentZmwIndex]);
                currentZmwIndex += p->zmwStride_;
            }
        }

        p->batchesSeen_++;
        std::unique_ptr<BazBuffer> ret;
        if (p->batchesSeen_ == p->numBatches_)
        {
            p->batchesSeen_ = 0;
            if (p->chunksSeen_ == p->chunksPerOutput_ - 1)
            {
                ret = std::make_unique<BazBuffer>(p->numZmw_, 0, p->expectedPulses,
                                                  CreateAllocator(AllocatorMode::MALLOC, SOURCE_MARKER()));
                std::swap(ret, buffer_);
            }
        }

        return ret;
    }

    std::unique_ptr<BazIO::BazBuffer> buffer_;
};

struct PrelimHQFilterBody::MultipleBuffer
{
    MultipleBuffer(PrelimHQFilterBody::Impl* p)
    {
        for (const auto& kv : p->poolDims_)
        {
            buffers_[kv.first] = std::make_unique<BazIO::BazBuffer>(kv.second.ZmwsPerBatch(), kv.first,
                                                                    PrelimHQFilterBody::Impl::expectedPulses,
                                                                    CreateAllocator(AllocatorMode::MALLOC, SOURCE_MARKER()));
        }
    }

    template <typename Serializer>
    std::unique_ptr<BazBuffer> Process(PrelimHQFilterBody::Impl* p, std::vector<Serializer>& serializers, BatchResult in)
    {
        const auto& pulseBatch = in.pulses;
        const auto& metricsPtr = in.metrics;

        // No need to worry about ordering if we are configured for multiple BAZ file but log information.
        if (pulseBatch.GetMeta().FirstFrame() > p->currFrame_)
        {
            if (p->batchesSeen_ != 0)
                PBLOG_INFO << "Data out of order, new chunk seen before all batches of previous chunk";
            p->currFrame_ = pulseBatch.GetMeta().FirstFrame();
            p->chunksSeen_++;
            if (p->chunksSeen_ == p->chunksPerOutput_) p->chunksSeen_ = 0;
        }
        else if (pulseBatch.GetMeta().FirstFrame() < p->currFrame_)
        {
            PBLOG_INFO << "Data out of order, multiple chunks being processed simultaneously";
        }

        size_t currentZmwIndex = 0;
        for (uint32_t lane = 0; lane < pulseBatch.Dims().lanesPerBatch; ++lane)
        {
            const auto& lanePulses = pulseBatch.Pulses().LaneView(lane);

            for (uint32_t zmw = 0; zmw < laneSize; zmw += p->zmwStride_)
            {
                if (metricsPtr)
                {
                    buffers_[pulseBatch.GetMeta().PoolId()]
                            ->AddMetrics(currentZmwIndex,
                                         [&](BazIO::MemoryBufferView<Primary::SpiderMetricBlock>& dest) {
                                             assert(dest.size() == 1);
                                             ConvertMetric(metricsPtr, dest[0], lane, zmw);
                                         },
                                         1);
                }
                auto pulses = lanePulses.ZmwData(zmw);
                buffers_[pulseBatch.GetMeta().PoolId()]
                        ->AddZmw(currentZmwIndex, pulses, pulses + lanePulses.size(zmw),
                                 serializers[currentZmwIndex]);
                currentZmwIndex += p->zmwStride_;
            }
        }

        p->batchesSeen_++;
        std::unique_ptr<BazBuffer> ret;
        // We should make a new buffer each time as the pool is complete.
        ret = std::make_unique<BazBuffer>(buffers_[pulseBatch.GetMeta().PoolId()]->NumZmw(), pulseBatch.GetMeta().PoolId(),
                                          PrelimHQFilterBody::Impl::expectedPulses,
                                          CreateAllocator(AllocatorMode::MALLOC, SOURCE_MARKER()));
        std::swap(ret, buffers_[pulseBatch.GetMeta().PoolId()]);
        return ret;
    }

    std::map<uint32_t, std::unique_ptr<BazIO::BazBuffer>> buffers_;
};

template <bool internal, bool multipleBazFiles>
struct PrelimHQFilterBody::ImplChild : public PrelimHQFilterBody::Impl
{
    ImplChild(size_t numZmws, const std::map<uint32_t, BatchDimensions>& poolDims,
              const PrelimHQConfig& config)
        : Impl(numZmws, poolDims, config)
        , serializers_(numZmws)
        , buffer_(this)
    { }

    std::unique_ptr<BazBuffer> Process(BatchResult in) override
    {
        return buffer_.Process(this, serializers_, std::move(in));
    }

    using Serializer = std::conditional_t<internal, InternalPulses, ProductionPulses>;
    std::vector<Serializer> serializers_;
    using BufferT = std::conditional_t<multipleBazFiles, MultipleBuffer, SingleBuffer>;
    BufferT buffer_;
};

PrelimHQFilterBody::PrelimHQFilterBody(size_t numZmws, const std::map<uint32_t, Data::BatchDimensions>& poolDims,
                                       const PrelimHQConfig& config, bool internal,
                                       bool multipleBazFiles)
{

    if (internal)
    {
        if (multipleBazFiles)
            impl_ = std::make_unique<ImplChild<true, true>>(numZmws, poolDims, config);
        else
            impl_ = std::make_unique<ImplChild<true, false>>(numZmws, poolDims, config);
    }
    else
    {
        if (multipleBazFiles)
            impl_ = std::make_unique<ImplChild<false, true>>(numZmws, poolDims, config);
        else
            impl_ = std::make_unique<ImplChild<false, false>>(numZmws, poolDims, config);
    }
}

PrelimHQFilterBody::~PrelimHQFilterBody() = default;

void PrelimHQFilterBody::Process(Mongo::Data::BatchResult in)
{
    auto ret = impl_->Process(std::move(in));
    if (ret) this->PushOut(std::move(ret));
}

}}
