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

#ifndef PACBIO_BAZIO_WRITING_BAZ_AGGREGATOR_H
#define PACBIO_BAZIO_WRITING_BAZ_AGGREGATOR_H

#include <memory>

#include <pacbio/datasource/MallocAllocator.h>

#include <bazio/MetricBlock.h>
#include <bazio/writing/BazBuffer.h>
#include <bazio/writing/PacketBuffer.h>
#include <bazio/writing/MetricBuffer.h>
#include <bazio/writing/MetricBufferManager.h>

namespace PacBio {
namespace BazIO {

/// Class that manages pulse and metric data as they incrementally stream in.
/// You can gather as much data as you like, and simply call `ProduceBazBuffer`
/// when you wish to grab what is currently available and hand it off for
/// writing to the baz file.
///
/// The BazAggregator has the concept of "HQ" vs "preHQ" zmw, where with
/// the latter we wish to accrue data, but maybe not output it to disk until
/// later when we are reasonably certain the data belongs to an HQ region. Data
/// for "preHQ" zmw are stored separately, in a vector of buffers where each
/// buffer corresponds to the interval between previous calls to ProduceBazBuffer.
/// There is a configurable max lookback, where once data has been around long
/// enough we drop it on the floor.
///
/// The BazAggregator provides an iterator interface to examine the preHQ data
/// and potentially mark is as now being "HQ".  When a ZMW is first marked as
/// HQ, then the next call to `ProduceBazBuffer` will expose the full lookback
/// data in the resulting BazBuffer.
template <typename MetricBlockT, typename AggregatedMetricBlockT>
class BazAggregator
{
public:
    using MetricBufferManagerT = MetricBufferManager<MetricBlockT,AggregatedMetricBlockT>;
public:
    /// Simplified constructor, predominantly meant for simulation/test execution where
    /// we don't care about HQ vs preHQ data.  All data is marked as HQ from the onset,
    /// and the lookback functionality is disabled.
    ///
    /// \param numZmw    The number of ZMW handled by this aggregator
    /// \param bufferId  The id associated with this particular aggregator
    /// \param expectedPulseBytesPerZmw The number of bytes to reserve for each ZMW
    ///                                 initially.  If this allotment is fully
    ///                                 consumed by any particular ZMW, then another
    ///                                 allocation of the same size will be reserved for it.
    BazAggregator(size_t numZmw, uint32_t bufferId,
                  size_t expectedPulseBytesPerZmw)
    : numZmw_(numZmw)
    , bufferId_(bufferId)
    , packetsAllocator_(std::make_shared<DataSource::MallocAllocator>())
    , packets_(numZmw, expectedPulseBytesPerZmw)
    , metricsAllocator_(std::make_shared<DataSource::MallocAllocator>())
    , metrics_(numZmw)
    {}

    /// Primary constructor, where all ZMW start off as "preHQ" until they are
    /// explicitly marked otherwise
    ///
    /// \param numZmw    The number of ZMW handled by this aggregator
    /// \param bufferId  The id associated with this particular aggregator
    /// \param expectedPulseBytesPerZmw The number of bytes to reserve for each zmw
    ///                                 initially.  If this allotment is fully
    ///                                 consumed by any particular ZMW, then
    ///                                 another allocation of the same size will
    ///                                 be reserved for it.
    /// \param maxLookback The number of data buffers we retain before dropping
    ///                    data Every call to `ProduceBazFile` will create a new
    ///                    data buffer
    /// \param packetsAllocator   An optional IAllocator instance to use for allocating
    ///                           packets memory
    /// \param metricsAllocatr    An optional IAllocator instance to use for allocating
    ///                           metrics memory
    BazAggregator(size_t numZmw,
                  uint32_t bufferId,
                  size_t expectedPulseBufferSize,
                  size_t maxLookback,
                  bool enableLookback,
                  std::shared_ptr<Memory::IAllocator> packetsAllocator =
                      std::make_shared<DataSource::MallocAllocator>(),
                  std::shared_ptr<Memory::IAllocator> metricsAllocator =
                      std::make_shared<DataSource::MallocAllocator>())
    : numZmw_(numZmw)
    , bufferId_(bufferId)
    , packetsAllocator_(packetsAllocator)
    , packets_(numZmw, expectedPulseBufferSize, maxLookback, packetsAllocator)
    , metricsAllocator_(metricsAllocator)
    , metrics_(numZmw, maxLookback, enableLookback, metricsAllocator)
    {}

    size_t NumZmw() const { return numZmw_; }
    uint32_t BufferId() const { return bufferId_; }

    /// Takes the currently accumulated data and produces a BazBuffer to write to disk.
    /// The new baz buffer takes full ownership of any data already marked as HQ, and this
    /// aggregator restarts them with a clean state.  The baz buffer also gets a shared_pointer
    /// reference to all the preHQ data, so that it can see (and extend the lifetime if necessary)
    /// all the data that wasn't HQ before, but recently has been marked as such.
    std::unique_ptr<BazBuffer> ProduceBazBuffer()
    {
        auto currPackets = packets_.CreateCheckpoint();
        auto currMetrics = metrics_.GetMetrics();
        auto ret = std::make_unique<BazBuffer>(bufferId_,
                                               std::move(currMetrics.first),
                                               std::move(currMetrics.second),
                                               std::move(currPackets));
        return ret;
    }

    // Flushes the current buffers of all remaining data.
    std::unique_ptr<BazBuffer> Flush()
    {
        auto currPackets = packets_.CreateCheckpoint();
        metrics_.Flush();
        auto currMetrics = metrics_.GetMetrics();
        auto ret = std::make_unique<BazBuffer>(bufferId_,
                                               std::move(currMetrics.first),
                                               std::move(currMetrics.second),
                                               std::move(currPackets));
        return ret;
    }

    /// Updates the lookback buffers for metrics. This is meant to be
    /// called at the metric block frequency.
    void UpdateLookback()
    {
        metrics_.UpdateLookback();
    }

    /// Adds ZMW packet data to the current buffers
    ///
    /// \param zmw   The zmw index of the data
    /// \param begin An interator to the first pulse to be serialized
    /// \param end   An iterator to one-past-the-end for the pulses to be serialized
    /// \predicate   A boolean functor to determine if a given pulse should be
    ///              serialized (e.g. to skip pulses not marked as bases)
    /// \serializer  A serializer that can convert the pulse to a byte stream
    template <typename Iterator, typename Predicate, typename Serializer>
    void AddPulses(size_t zmw,
                   Iterator begin,
                   Iterator end,
                   Predicate&& predicate,
                   Serializer&& serializer)
    {
        packets_.AddZmw(zmw,
                        begin,
                        end,
                        std::forward<Predicate>(predicate),
                        std::forward<Serializer>(serializer));
    }

    /// Adds metric data to the current buffers
    ///
    /// \param zmw   The zmw index of the data
    /// \param mb    The metrics converted to metric block format
    void AddMetrics(size_t zmw, const std::vector<MetricBlockT>& mb)
    {
        metrics_.AddZmw(zmw, mb);
    }

    ~BazAggregator() = default;

    /// Iterator interface used to iterate over "preHQ" data only,
    /// with the intent of deciding if said data should now be
    /// marked as HQ.
    struct PreHQIterator
    {
        PreHQIterator(size_t idx,
                      PacketBufferManager& packets,
                      MetricBufferManagerT& metrics)
            : idx_(idx)
            , packets_(packets)
            , metrics_(metrics)
        {}
        // bare minimum to allow range based for loops...
        bool operator!=(const PreHQIterator& o) const
        {
            assert(std::addressof(packets_.get()) == std::addressof(o.packets_.get()));
            return idx_ != o.idx_;
        }
        PreHQIterator& operator++()
        {
            assert(idx_ < packets_.get().NumZmw());
            do
            {
                idx_++;
            } while (idx_ < packets_.get().NumZmw() && packets_.get().IsHQ(idx_));
            return *this;
        }
        struct DerefView
        {
            DerefView(PreHQIterator* itr)
                : itr_(itr)
            {}

            void MarkAsHQ()
            {
                itr_->packets_.get().MarkAsHQ(itr_->idx_);
                itr_->metrics_.get().MarkAsHQ(itr_->idx_);
            }

            const Mongo::Data::HQRFPhysicalStates GetRecentActivityLabel()
            {
                return itr_->metrics_.get().GetRecentActivityLabel(itr_->idx_);
            }
        private:
            PreHQIterator* itr_;
        };
        DerefView operator*() { return DerefView{this}; }

    private:
        size_t idx_;
        std::reference_wrapper<PacketBufferManager> packets_;
        std::reference_wrapper<MetricBufferManagerT> metrics_;
    };

    // Wrapper struct, whose sole purpose is to provide the
    // begin/end function necessary for range based for loops
    struct PreHQView
    {
        PreHQView(PacketBufferManager& packets, MetricBufferManagerT& metrics)
            : packets_(packets)
            , metrics_(metrics)
        {}
        PreHQIterator begin()
        {
            PreHQIterator ret{0, packets_, metrics_};
            if (packets_.get().IsHQ(0)) ++ret;
            return ret;
        }
        PreHQIterator end()
        {
            return PreHQIterator{packets_.get().NumZmw(), packets_, metrics_};
        }
        size_t size() const
        {
            return packets_.get().NumZmw() - packets_.get().NumHQ();
        }

    private:
        std::reference_wrapper<PacketBufferManager> packets_;
        std::reference_wrapper<MetricBufferManagerT> metrics_;
    };

    PreHQView PreHQData()
    {
        return PreHQView{packets_, metrics_};
    }

private:
    size_t numZmw_;
    uint32_t bufferId_;

    std::shared_ptr<Memory::IAllocator> packetsAllocator_;
    PacketBufferManager packets_;

    std::shared_ptr<Memory::IAllocator> metricsAllocator_;
    MetricBufferManagerT metrics_;
};


}}

#endif //PACBIO_BAZIO_WRITING_BAZ_AGGREGATOR_H
