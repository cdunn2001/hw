// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#ifndef PACBIO_METRICIO_WRITING_METRIC_BUFFER_MANAGER_H
#define PACBIO_METRICIO_WRITING_METRIC_BUFFER_MANAGER_H


#include <bazio/writing/MemoryBuffer.h>
#include <bazio/writing/MetricBlock.h>
#include <bazio/writing/MetricBuffer.h>
#include <dataTypes/HQRFPhysicalStates.h>

namespace PacBio::BazIO {

/// Class used to managed metrics as they are streamed in for a collection
/// of ZMWs. There are two modes the metrics buffer manager operates in:
///
/// Production mode:
/// A lookback buffer of production metrics is maintained for the preHQ
/// algorithm. Metric blocks are popped off the end of the lookback buffer
/// and aggregated to a pre-PreHQ metric block until the preHQ algorithm
/// has detected a preHQ region. Once the preHQ algorithm has determined
/// a preHQ region has started, the aggregated pre-PreHQ metric block
/// is written to disk and the contents of the lookback buffer (aggregated
/// by activity label) are written to disk. From there, metric blocks
/// are aggregated based on activity label and written to disk.
///
/// Internal mode:
/// Internal metrics are not aggregated and written at the metric block
/// frequency level. Given the I/O load for this mode, this is meant
/// to be run off-line and on a subset of the chip for both preHQ
/// and HQRF algorithm training purposes.
///
/// \tparam MetricBlockT            The metric block type that inherits from the MetricBlock CRTP
/// \tparam AggregatedMetricBlockT  The aggregated metric block type that inherits from the MetricBlock CRTP
template <typename MetricBlockT, typename AggregatedMetricBlockT>
class MetricBufferManager
{
public:
    using MetricBufferT = MetricBuffer<MetricBlockT>;
    using AggregatedMetricBufferT = MetricBuffer<AggregatedMetricBlockT>;
    using OutputMetricT = Primary::SpiderMetricBlock;
public:
    // Simplified constructor for simulation/testing
    MetricBufferManager(size_t numZmw)
    : allocator_(std::make_shared<DataSource::MallocAllocator>())
    , mostRecentActivityLabels_(numZmw, Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES)
    , indexInfo_(numZmw)
    , newHQ_(numZmw, true)
    , numHQ_(numZmw)
    , maxLookBack_(0)
    , metricsBuffer_(std::max(indexInfo_.size(), 100ul), 1, *allocator_)
    , metrics_(numZmw)
    {
        for (size_t zmw = 0; zmw < numZmw; ++zmw)
        {
            indexInfo_[zmw].hqStarted_ = true;
            indexInfo_[zmw].currentIndex_ = zmw;
            indexInfo_[zmw].recentIndex_ = zmw;
            aggregatedMetrics_->Data(zmw)->Reset();
        }
    }

    // Constructor
    MetricBufferManager(size_t numZmw, size_t maxLookBack, bool enableLookback,
                        std::shared_ptr<Memory::IAllocator> allocator)
    : allocator_(allocator)
    , mostRecentActivityLabels_(numZmw, Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES)
    , indexInfo_(numZmw)
    , newHQ_(numZmw, false)
    , numHQ_(0)
    , aggregatedMetrics_(std::make_unique<AggregatedMetricBufferT>(numZmw, allocator_))
    , maxLookBack_(maxLookBack)
    , metricsBuffer_(std::max(indexInfo_.size(), 100ul), 1, *allocator_)
    , metrics_(numZmw)
    {
        if (enableLookback)
        {
            lookbackWindow_.push_back(std::make_shared<MetricBufferT>(numZmw, allocator_));
        }

        for (size_t zmw = 0; zmw < numZmw; ++zmw)
        {
            indexInfo_[zmw].hqStarted_ = false;
            indexInfo_[zmw].currentIndex_ = zmw;
            indexInfo_[zmw].recentIndex_ = zmw;
            aggregatedMetrics_->Data(zmw)->Reset();
        }
    }

public:
    /// \return The number of ZMW belonging to this manager.
    size_t NumZmw() const { return indexInfo_.size(); }

    /// \return The metrics to be written to be written.
    const MemoryBufferView<OutputMetricT>& GetMetrics(size_t zmw) const { return metrics_[zmw]; }

    /// \return The most recent activity label for the given zmw.
    Mongo::Data::HQRFPhysicalStates GetRecentActivityLabel(size_t zmw) const { return mostRecentActivityLabels_[zmw]; }

public:
    void AddZmw(size_t zmwIndex, const MetricBlockT& metrics)
    {
        mostRecentActivityLabels_[zmwIndex] = static_cast<Mongo::Data::HQRFPhysicalStates>(metrics.ActivityLabel());
        if (lookbackWindow_.empty())
            AddInternalMetrics(zmwIndex, metrics);
        else
            AddProductionMetrics(zmwIndex, metrics);
    }

    void MarkAsHQ(size_t zmw)
    {
        if (lookbackWindow_.empty()) return;

        if (indexInfo_[zmw].hqStarted_) return;
        if (newHQ_[zmw]) return;
        numHQ_++;
        newHQ_[zmw] = true;
    }

    std::unique_ptr<const MetricBufferManager> CreateCheckpoint()
    {
        // Copies the contents of our current instance, with the exception of metricsBuffer_
        // and metrics_ that gets destructively moved.
        auto ret = std::unique_ptr<MetricBufferManager>(new MetricBufferManager(*this));

        metricsBuffer_ = MemoryBuffer<OutputMetricT>(std::max(indexInfo_.size(), 100ul), 1, *allocator_);
        metrics_ = std::vector<MemoryBufferView<OutputMetricT>>(indexInfo_.size());

        return ret;
    }

    std::unique_ptr<const MetricBufferManager> Flush()
    {
        if (!lookbackWindow_.empty())
        {
            for (size_t zmw = 0; zmw < indexInfo_.size(); zmw++)
            {
                auto& aggregatedMb = *aggregatedMetrics_->Data(zmw);
                if (aggregatedMb.HasData())
                {
                    auto& metricsOut = metrics_[zmw];
                    metricsOut = metricsBuffer_.Allocate(1);
                    aggregatedMb.Convert(metricsOut[0]);
                }
            }
        }

        auto ret = std::unique_ptr<MetricBufferManager>(new MetricBufferManager(*this));
        return ret;
    }

    // Updates the lookback buffer, meant to be called at metric block intervals.
    void UpdateLookback()
    {
        if (lookbackWindow_.empty()) return;

        size_t lookbackIdx = 0;
        for (size_t zmw = 0; zmw < indexInfo_.size(); zmw++)
        {
            if (indexInfo_[zmw].hqStarted_ || newHQ_[zmw])
            {
                indexInfo_[zmw].hqStarted_ = true;
                if (newHQ_[zmw])
                {
                    // Newly created HQ-region.
                    size_t numActivityLabels = NumActivityLabels(zmw);
                    auto& metricsOut = metrics_[zmw];
                    metricsOut = metricsBuffer_.Allocate(numActivityLabels + 1);

                    // Add preHQ aggregated metric block to output buffer.
                    auto& aggregatedMb = *aggregatedMetrics_->Data(zmw);
                    aggregatedMb.Convert(metricsOut[0]);
                    aggregatedMb.Reset();

                    // Go through lookback window, aggregate and add to output buffer.
                    AddLookbackWindowMetricBlocks(zmw);
                }
            }
            else
            {
                lookbackWindow_.back()->Link(indexInfo_[zmw].recentIndex_, lookbackIdx);
                indexInfo_[zmw].recentIndex_ = lookbackIdx;
                lookbackIdx++;

                if (lookbackWindow_.size() == maxLookBack_)
                {
                    // Take metric block from front of buffer and aggregate to preHQ metric block.
                    const auto& frontMetricBlock = *lookbackWindow_.front()->Data(indexInfo_[zmw].currentIndex_);
                    auto& aggregatedMb = *aggregatedMetrics_->Data(zmw);
                    aggregatedMb.Aggregate(frontMetricBlock);
                    indexInfo_[zmw].currentIndex_ = *lookbackWindow_.front()->Index(indexInfo_[zmw].currentIndex_);
                }
            }
        }

        if (maxLookBack_ > 0)
        {
            // Create new metric block buffer only for those ZMWs where the preHQ has not started.
            lookbackWindow_.push_back(std::make_shared<MetricBufferT>(indexInfo_.size() - numHQ_, allocator_));

            if (lookbackWindow_.size() > maxLookBack_)
                lookbackWindow_.erase(lookbackWindow_.begin());
        }

        newHQ_ = std::vector<bool>(indexInfo_.size(), false);
    }

private:

    // Copy constructor where the metrics are "moved".
    MetricBufferManager(MetricBufferManager& other)
    : allocator_(other.allocator_)
    , mostRecentActivityLabels_(other.mostRecentActivityLabels_)
    , indexInfo_(other.indexInfo_)
    , newHQ_(other.newHQ_)
    , numHQ_(other.numHQ_)
    , maxLookBack_(other.maxLookBack_)
    , lookbackWindow_(other.lookbackWindow_)
    // These are moved.
    , metricsBuffer_(std::move(other.metricsBuffer_))
    , metrics_(std::move(other.metrics_))
    {}

    void AddInternalMetrics(size_t zmwIndex, const MetricBlockT& metrics)
    {
        auto& metricsOut = metrics_[zmwIndex];
        metricsOut = metricsBuffer_.Allocate(1);
        metrics.Convert(metricsOut[0]);
    }

    void AddProductionMetrics(size_t zmwIndex, const MetricBlockT& metrics)
    {
        if (!indexInfo_[zmwIndex].hqStarted_)
        {
            // Haven't started the preHQ region, add to the last
            // (most recent) buffer in the lookback window.
            lookbackWindow_.back()->Data(indexInfo_[zmwIndex].recentIndex_)->Set(metrics);
        }
        else
        {
            // PreHQ region has already started, aggregate.
            auto& aggregatedMb = *aggregatedMetrics_->Data(indexInfo_[zmwIndex].currentIndex_);
            if (metrics.ActivityLabel() == aggregatedMb.ActivityLabel())
            {
                // Activity labels match, aggregate.
                aggregatedMb.Aggregate(metrics);
            }
            else
            {
                // Activity labels don't match.
                if (static_cast<uint8_t>(Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES) !=
                    aggregatedMb.ActivityLabel())
                {
                    // Add to output buffer.
                    auto& metricsOut = metrics_[zmwIndex];
                    metricsOut = metricsBuffer_.Allocate(1);
                    aggregatedMb.Convert(metricsOut[0]);
                }

                aggregatedMb.Set(metrics);
            }
        }
    }

    // Goes through the lookback buffer and counts the number of
    // different contiguous activity labels.
    size_t NumActivityLabels(size_t zmw)
    {
        uint32_t currentIndex = indexInfo_[zmw].currentIndex_;
        size_t numActivityLabels = 0;
        uint8_t currActivityLabel = static_cast<uint8_t>(Mongo::Data::HQRFPhysicalStates::NUM_PHYS_STATES);
        for (const auto& metricBuffer : lookbackWindow_)
        {
            const auto& metricBlock = *metricBuffer->Data(currentIndex);
            if (metricBlock.ActivityLabel() != currActivityLabel)
            {
                numActivityLabels++;
                currActivityLabel = metricBlock.ActivityLabel();
            }
            currentIndex = *metricBuffer->Index(currentIndex);
        }
        return numActivityLabels;
    }

    void AddLookbackWindowMetricBlocks(size_t zmw)
    {
        uint32_t currentIndex = indexInfo_[zmw].currentIndex_;
        const auto& frontMb = *lookbackWindow_.front()->Data(currentIndex);
        uint8_t currActivityLabel = frontMb.ActivityLabel();
        AggregatedMetricBlockT aggregatedMb;
        aggregatedMb.Set(frontMb);
        size_t outputBlock = 1;
        auto& metricsOut = metrics_[zmw];
        for (auto metricBuffer = lookbackWindow_.begin()+1; metricBuffer != lookbackWindow_.end(); metricBuffer++)
        {
            const auto& mb = *(*metricBuffer)->Data(currentIndex);
            if (mb.ActivityLabel() != currActivityLabel)
            {
                aggregatedMb.Convert(metricsOut[outputBlock]);
                outputBlock++;
                currActivityLabel = mb.ActivityLabel();
                aggregatedMb.Set(mb);
            }
            else
            {
                aggregatedMb.Aggregate(mb);
            }
            currentIndex = *(*metricBuffer)->Index(currentIndex);
        }
        aggregatedMb.Convert(metricsOut[outputBlock]);
    }

private:
    struct IndexInfo
    {
        bool hqStarted_;
        uint32_t currentIndex_;
        uint32_t recentIndex_;
    };
private:
    std::shared_ptr<Memory::IAllocator> allocator_;
    std::vector<Mongo::Data::HQRFPhysicalStates> mostRecentActivityLabels_;

    std::vector<IndexInfo> indexInfo_;
    std::vector<bool> newHQ_;
    size_t numHQ_;

    std::unique_ptr<AggregatedMetricBufferT> aggregatedMetrics_;

    size_t maxLookBack_;
    std::vector<std::shared_ptr<MetricBufferT>> lookbackWindow_;

    MemoryBuffer<OutputMetricT> metricsBuffer_;
    std::vector<MemoryBufferView<OutputMetricT>> metrics_;
};

} // namespace PacBio::BazIO

#endif // PACBIO_METRICIO_WRITING_METRIC_BUFFER_MANAGER_H