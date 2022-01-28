// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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

#pragma once

#include <memory>
#include <unordered_map>

#include "BlockLevelMetrics.h"

#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/EncodingParams.h>
#include <bazio/file/FileHeader.h>

namespace PacBio {
namespace Primary {

struct ZmwDataCounts
{
    size_t packetsByteSize = 0;
    size_t numMBs = 0;
};

// Class used to aggregate the raw binary data for a given zmw.
class ZmwByteData
{
public:
    // Represents a fixed size memory buffer for binary data. Provides an
    // interface for incremental streaming writes to full the buffer, and
    // const access to the full data (intended to be used after it is filled,
    // though the API does not enforce that)
    class ByteStream
    {
    public:
        ByteStream(size_t capacity)
          : data_(new uint8_t[capacity])
          , size_(0)
          , capacity_(capacity)
        {}

        size_t capacity() const { return capacity_; }
        size_t size() const { return size_; }
        bool IsFull() const { return size_ == capacity_; }

        // Returns a pointer to the next `count` bytes, for calling code to
        // fill with data.  If there is not enough unused space to satisfy
        // the requested `count`, a nullptr is returned
        uint8_t* next(size_t count)
        {
            if (size_ + count > capacity_) return nullptr;

            uint8_t* ret = data_.get() + size_;
            size_ += count;
            return ret;
        }
        // const access to the full data.  Only really meant to be called after
        // the raw data is filled, though if that changes feel free to remove
        // the assert.

        const uint8_t* get() const
        {
            assert(size_ == capacity_);
            return data_.get();
        }

    private:
        std::unique_ptr<uint8_t[]> data_;
        size_t size_;
        size_t capacity_;
    };

    ZmwByteData(uint32_t metricByteSize, const ZmwDataCounts& expectedSizes, size_t idx)
      : packetByteStream_(expectedSizes.packetsByteSize)
      , metricByteStream_(expectedSizes.numMBs * metricByteSize)
      , sizes_(expectedSizes)
      , numEvents_(0)
      , zmwIndex_(idx)
    {}

    // Functions to return the next unused bytes for incremental filling of
    // the data.  If more memory is requested than was initially allocated,
    // these functions will return nullptr
    uint8_t * NextPacketBytes(size_t size, size_t numEvents)
    {
        numEvents_ += numEvents;
        return packetByteStream_.next(size);
    }
    uint8_t * NextMetricBytes(size_t size)
    {
        return metricByteStream_.next(size);
    }

    size_t NumEvents() const { return numEvents_; }

    const ZmwDataCounts& Sizes() const { return sizes_; }

    size_t NumBytes() const
    {
        return packetByteStream_.size() + metricByteStream_.size();
    }

    bool IsFull() const
    {
        return packetByteStream_.IsFull() && metricByteStream_.IsFull();
    }

    const ByteStream& packetByteStream() const
    {
        return packetByteStream_;
    }
    const ByteStream& MetricByteStream() const
    {
        return metricByteStream_;
    }

    size_t ZmwIndex() const { return zmwIndex_; }

private:
    ByteStream packetByteStream_;
    ByteStream metricByteStream_;

    ZmwDataCounts sizes_;
    size_t numEvents_;

    size_t zmwIndex_;
};

// Provides a raw read-only API to directly access the data parsed from the
// Baz file.  It will have been deserialized into a vector of uint32_t values,
// but any transformations done to the data upstream will not yet have been
// reverted
class RawEventData
{
    using PulseFieldParams = BazIO::FieldParams<BazIO::PacketFieldName>;
public:
    // Accepts the vector of vectors for the parsed packet data, as produced
    // where each entry into the outermost vector corresponds to a given
    // PacketFieldName.
    RawEventData(std::vector<std::vector<uint32_t>>&& rawPacketData,
                 const std::vector<PulseFieldParams>& fieldInfo)
        : data_(std::move(rawPacketData))
        , fieldInfo_(fieldInfo)
    {
        numEvents_ = 0;
        for (const auto& packet : data_)
        {
            if (packet.size() > 0)
            {
                numEvents_ = packet.size();
                break;
            }
        }
        #ifndef NDEBUG
        for (const auto& packet : data_)
        {
            assert(packet.size() == 0 || packet.size() == numEvents_);
        }
        #endif
    }

    // Move only semantics
    RawEventData(const RawEventData&) = delete;
    RawEventData(RawEventData&&) = default;
    RawEventData& operator=(const RawEventData&) = delete;
    RawEventData& operator=(RawEventData&&) = default;

public:  // const data accessors
    size_t NumEvents() const { return numEvents_; }

    bool HasPacketField(BazIO::PacketFieldName name) const
    {
        return data_[static_cast<uint32_t>(name)].size() > 0;
    };
    const std::vector<uint32_t>& PacketField(BazIO::PacketFieldName name) const
    {
        return data_[static_cast<uint32_t>(name)];
    };

    const std::vector<PulseFieldParams>& FieldInfo() const
    {
        return fieldInfo_;
    };

private:
    std::vector<std::vector<uint32_t>> data_;
    size_t numEvents_;
    std::vector<PulseFieldParams> fieldInfo_;
};

class RawMetricData
{
public:
    RawMetricData() = default;

    RawMetricData(const std::vector<MetricField>& fields)
    {
        size_t numIntMetrics = 0;
        size_t numUIntMetrics = 0;
        size_t numFloatMetrics = 0;

        for (const auto& f : fields)
        {
            switch(f.fieldBitSize)
            {
                case 8:
                case 16:
                case 32:
                    if (f.fieldScalingFactor != 1)
                    {
                        whichVec_[f.fieldName] = numFloatMetrics;
                        numFloatMetrics++;
                    }
                    else
                    {
                        if (f.fieldSigned)
                        {
                            whichVec_[f.fieldName] = numIntMetrics;
                            numIntMetrics++;
                        }
                        else
                        {
                            whichVec_[f.fieldName] = numUIntMetrics++;
                            numUIntMetrics++;
                        }
                    }
                    break;
                default:
                    throw PBException(
                            "Didn't expect that field bit-size: " +
                            std::to_string(f.fieldBitSize));
            }
        }

        idata_.resize(numIntMetrics);
        udata_.resize(numUIntMetrics);
        fdata_.resize(numFloatMetrics);
    }

    // Move only semantics
    RawMetricData(const RawMetricData&) = delete;
    RawMetricData(RawMetricData&&) = default;
    RawMetricData& operator=(const RawMetricData&) = delete;
    RawMetricData& operator=(RawMetricData&&) = default;
public:

    size_t ParseMetricField(const MetricField& f, const uint8_t* buffer);

    bool HasMetric(const MetricFieldName& f) const
    {
        return whichVec_.find(f) != whichVec_.end();
    }

    bool Empty() const
    {
        // We consider metrics empty only if all of them are.
        for (const auto& m : idata_) if (!m.empty()) return false;
        for (const auto& m : udata_) if (!m.empty()) return false;
        for (const auto& m : fdata_) if (!m.empty()) return false;

        return true;
    }

    auto IntMetric(const MetricFieldName& f) const
    {
        return idata_[whichVec_.at(f)];
    }

    auto UIntMetric(const MetricFieldName& f) const
    {
        return udata_[whichVec_.at(f)];
    }

    auto FloatMetric(const MetricFieldName& f) const
    {
        return fdata_[whichVec_.at(f)];
    }

    const auto& IntMetrics() const
    {
        return idata_;
    }

    const auto& UIntMetrics() const
    {
        return udata_;
    }

    const auto& FloatMetrics() const
    {
        return fdata_;
    }

    // The below are provided for testing purposes.

    auto& IntMetric(const MetricFieldName& f)
    {
        return idata_[whichVec_.at(f)];
    }

    auto& UIntMetric(const MetricFieldName& f)
    {
        return udata_[whichVec_.at(f)];
    }

    auto& FloatMetric(const MetricFieldName& f)
    {
        return fdata_[whichVec_.at(f)];
    }

private:
    template<typename T>
    void ReadFixedPointFromBaz(const MetricField& f, const uint8_t* buffer);

    template<typename T>
    void ReadIntFromBaz(const MetricField& f, const uint8_t* buffer);

    template<typename T>
    void ReadUIntFromBaz(const MetricField& f, const uint8_t* buffer);

    void ReadHalfFloatFromBaz(const MetricField& f, const uint8_t* buffer);

private:
    std::unordered_map<size_t,size_t>   whichVec_;
    std::vector<std::vector<int32_t>>   idata_;
    std::vector<std::vector<uint32_t>>  udata_;
    std::vector<std::vector<float>>     fdata_;
};

// Free functions, used to help parse binary data into metrics and packet
// information

BlockLevelMetrics ParseMetrics(const std::vector<MetricField>& metricFields,
                               uint32_t metricFrames,
                               double frameRateHz,
                               const std::vector<float> relAmps,
                               const std::string& baseMap,
                               const ZmwByteData& data, bool internal);
RawEventData ParsePackets(const std::vector<BazIO::GroupParams<BazIO::PacketFieldName>>& packetGroups,
                          const std::vector<BazIO::FieldParams<BazIO::PacketFieldName>>& packetFields,
                          const ZmwByteData& data);
RawEventData ParsePackets(const std::vector<BazIO::GroupParams<BazIO::PacketFieldName>>& groups,
                          const uint8_t* data,
                          size_t dataSize,
                          size_t numEvents);
RawMetricData ParseMetricFields(const std::vector<MetricField>& fields, const ZmwByteData::ByteStream& data);

}}
