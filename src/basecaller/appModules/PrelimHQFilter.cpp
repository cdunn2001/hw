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

#include <appModules/PrelimHQFilter.h>

#include <bazio/MetricBlock.h>
#include <bazio/writing/BazAggregator.h>
#include <bazio/writing/BazBuffer.h>
#include <bazio/writing/MetricBlock.h>
#include <pacbio/smrtdata/Basecall.h>

#include <dataTypes/configs/PrelimHQConfig.h>
#include <dataTypes/configs/SmrtBasecallerConfig.h>
#include <dataTypes/configs/SystemsConfig.h>
#include <dataTypes/Metrics.h>
#include <dataTypes/PulseGroups.h>
#include <dataTypes/PulseFieldAccessors.h>

using namespace PacBio::Mongo::Data;
using namespace PacBio::Mongo;
using namespace PacBio::BazIO;

namespace PacBio {
namespace Application {

using namespace Cuda::Memory;
using namespace PacBio::BazIO;
using namespace PacBio::Mongo::Data;

class PrelimHQFilterBody::Impl
{
public:

    Impl() = default;
    virtual ~Impl() = default;

    virtual std::unique_ptr<BazBuffer> Process(BatchResult in) = 0;
    virtual std::unique_ptr<BazBuffer> Flush() = 0;
};

template <bool internal>
class PrelimHQFilterBody::ImplChild : public PrelimHQFilterBody::Impl
{
private:
    static_assert(InternalPulses::nominalBytes == 6);
    static_assert(ProductionPulses::nominalBytes == 2);
    static constexpr size_t expectedBytesPerBase = internal
            ? InternalPulses::nominalBytes : ProductionPulses::nominalBytes;
    using Serializer = std::conditional_t<internal, InternalPulses, ProductionPulses>;
    using BazAggregatorT = std::conditional_t<internal,
                            BazAggregator<CompleteMetricsGroup::MetricT,
                                          CompleteMetricsGroup::MetricAggregatedT>,
                            BazAggregator<ProductionMetricsGroup::MetricT,
                                          ProductionMetricsGroup::MetricAggregatedT>>;
public:
    ImplChild(size_t numZmws, size_t numBatches, uint32_t chunksPerMetric, uint32_t bufferId,
              bool multipleBazFiles, const PrelimHQConfig& config)
        : numBatches_(numBatches)
        , numZmw_(numZmws)
        , chunksPerMetric_(chunksPerMetric)
        , bazMetricBlocksPerOutput_(config.bazBufferMetricBlocks)
        , zmwStride_(config.zmwOutputStride)
        , multipleBazFiles_(multipleBazFiles)
        , aggregator_(std::make_unique<BazAggregatorT>(numZmws, bufferId,
                                                       config.expectedPulsesPerZmw*expectedBytesPerBase,
                                                       config.lookbackSize,
                                                       config.enablePreHQ,
                                                       CreateMallocAllocator(std::string("PrelimHQFilter_packetsAllocator"),
                                                                             CacheMode::GLOBAL_CACHE),
                                                       CreateMallocAllocator(std::string("PrelimHQFilter_metricsAllocator"),
                                                                             CacheMode::GLOBAL_CACHE)))
        , dummyPreHQ_(config)
        , serializers_(numZmws)
    {}

    std::unique_ptr<BazBuffer> Process(BatchResult in) override
    {
        const auto& pulseBatch = in.pulses;
        const auto& metricsPtr = in.metrics;

        if (pulseBatch.GetMeta().FirstFrame() > currFrame_)
        {
            if (batchesSeen_ % numBatches_ != 0)
                throw PBException("Data out of order, new chunk seen before all batches of previous chunk");
            currFrame_ = pulseBatch.GetMeta().FirstFrame();
        } else if (pulseBatch.GetMeta().FirstFrame() < currFrame_)
        {
            throw PBException("Data out of order, multiple chunks being processed simultaneously");
        }

        size_t currentZmwIndex = (multipleBazFiles_) ? 0 : pulseBatch.GetMeta().FirstZmw();
        for (uint32_t lane = 0; lane < pulseBatch.Dims().lanesPerBatch; ++lane)
        {
            const auto& lanePulses = pulseBatch.Pulses().LaneView(lane);
            for (uint32_t zmw = 0; zmw < laneSize; zmw += zmwStride_)
            {
                if (metricsPtr)
                {
                    const auto& metrics = metricsPtr->GetHostView()[lane];
                    aggregator_->AddMetrics(currentZmwIndex, { {metrics, zmw} });
                }
                auto pulses = lanePulses.ZmwData(zmw);
                aggregator_->AddPulses(currentZmwIndex,
                                       pulses,pulses + lanePulses.size(zmw),
                                       [](const auto& p){ return internal ? true : !p.IsReject(); },
                                       serializers_[currentZmwIndex]);
                currentZmwIndex += zmwStride_;
            }
        }
        batchesSeen_++;
        if (batchesSeen_ % (numBatches_ * chunksPerMetric_) == 0)
        {
            dummyPreHQ_.DetectHQ(*aggregator_);
            aggregator_->UpdateLookback();
        }
        if (batchesSeen_ == numBatches_ * chunksPerMetric_ * bazMetricBlocksPerOutput_)
        {
            batchesSeen_ = 0;
            return aggregator_->ProduceBazBuffer();
        }
        return std::unique_ptr<BazBuffer>{};
    }

    std::unique_ptr<BazBuffer> Flush() override
    {
        return aggregator_->Flush();
    }

private:
    int32_t currFrame_ = std::numeric_limits<int32_t>::min();
    size_t batchesSeen_ = 0;
    size_t numBatches_;
    size_t numZmw_;
    size_t chunksPerMetric_;
    size_t bazMetricBlocksPerOutput_;
    size_t zmwStride_;
    bool multipleBazFiles_;
    std::unique_ptr<BazAggregatorT> aggregator_;

    // Dummy class to serve as basic placeholder for the preliminary HQ algorithm.
    // It can be totally burnt to the ground once the real thing is about read.
    struct DummyPreHQ
    {
        DummyPreHQ(const PrelimHQConfig& config)
            : config_(config)
        {
            stride = static_cast<size_t>(std::round(1/config_.hqThrottleFraction));
        }

        void DetectHQ(BazAggregatorT& agg)
        {
            if (!config_.enablePreHQ)
            {
                for (auto&& v : agg.PreHQData()) v.MarkAsHQ();
                return;
            }

            seen++;
            if (seen < config_.lookbackSize) 
            {
                for (auto&& v : agg.PreHQData())
                    AddActivityLabel(v.GetRecentActivityLabel());
                return;
            }

            size_t counter = 0;
            size_t marked = 0;

            for (auto&& v : agg.PreHQData())
            {
                if (counter % stride == 0)
                {
                    if (HasPreHQStarted(v.GetRecentActivityLabel()))
                    {
                        marked++;
                        v.MarkAsHQ();
                    }
                }
                counter++;
            }
            PBLOG_DEBUG << "Enabled " << marked << " ZMW";
        }

        void AddActivityLabel(const Mongo::Data::HQRFPhysicalStates& state)
        {
            
        }

        bool HasPreHQStarted(const Mongo::Data::HQRFPhysicalStates& state)
        {
            return true;
        }

    private:
        PrelimHQConfig config_;

        size_t seen = 0;
        size_t stride;
    } dummyPreHQ_;

    std::vector<Serializer> serializers_;
};

PrelimHQFilterBody::PrelimHQFilterBody(
        size_t numZmws, const std::map<uint32_t,
        Data::BatchDimensions>& poolDims,
        const SmrtBasecallerConfig& config)
    : numThreads_(config.system.ioConcurrency)
{
    const uint32_t framesPerChunk = poolDims.begin()->second.framesPerBatch;
    uint32_t chunksPerMetric = (config.algorithm.Metrics.framesPerHFMetricBlock + framesPerChunk - 1)/framesPerChunk;
    if (config.multipleBazFiles)
    {
        for (const auto& kv : poolDims)
        {
            if (config.internalMode)
                impl_.push_back(std::make_unique<ImplChild<true>>(kv.second.ZmwsPerBatch(), 1,
                                                                  chunksPerMetric, kv.first,
                                                                  config.multipleBazFiles,
                                                                  config.prelimHQ));
            else
                impl_.push_back(std::make_unique<ImplChild<false>>(kv.second.ZmwsPerBatch(), 1,
                                                                   chunksPerMetric, kv.first,
                                                                   config.multipleBazFiles,
                                                                   config.prelimHQ));
        }
    }
    else
    {
        if (config.internalMode)
            impl_.push_back(std::make_unique<ImplChild<true>>(numZmws, poolDims.size(),
                                                              chunksPerMetric, 0,
                                                              config.multipleBazFiles, config.prelimHQ));
        else
            impl_.push_back(std::make_unique<ImplChild<false>>(numZmws, poolDims.size(),
                                                               chunksPerMetric, 0,
                                                               config.multipleBazFiles, config.prelimHQ));
    }
}

PrelimHQFilterBody::~PrelimHQFilterBody() = default;

void PrelimHQFilterBody::Process(Mongo::Data::BatchResult in)
{
    auto ret = (impl_.size() == 1)
            ? impl_[0]->Process(std::move(in))
            : impl_[in.pulses.GetMeta().PoolId()]->Process(std::move(in));
    if (ret) this->PushOut(std::move(ret));
}

std::vector<uint32_t> PrelimHQFilterBody::GetFlushTokens()
{
    std::vector<uint32_t> flushTokens;
    flushTokens.resize(impl_.size());
    std::iota(flushTokens.begin(), flushTokens.end(), 0);
    return flushTokens;
}

void PrelimHQFilterBody::Flush(uint32_t token)
{
    auto ret = impl_[token]->Flush();
    if (ret) this->PushOut(std::move(ret));
}

}}
