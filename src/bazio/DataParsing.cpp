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

#include "DataParsing.h"

#include <cstring>
#include <half.hpp>

#include <pacbio/logging/Logger.h>
#include <pacbio/smrtdata/QvCore.h>


#include "FloatFixedCodec.h"

#include <bazio/encoding/FieldNames.h>
#include <bazio/encoding/FieldSerializers.h>
#include <bazio/encoding/EncodingParams.h>

#include <common/utility/Overload.h>

using namespace PacBio::SmrtData::QvCore;
using namespace PacBio::BazIO;

namespace PacBio {
namespace Primary {

namespace
{

const size_t NUM_PACKET_FIELDS = BazIO::PacketFieldName::allValues().size();

std::vector<std::vector<uint32_t>> ParsePacketFields(
        const std::vector<BazIO::GroupParams<BazIO::PacketFieldName>>& encoding,
        const uint8_t * rawData,
        size_t rawDataSize,
        size_t numEvents)
{
    std::vector<std::vector<uint32_t>> packetFields(NUM_PACKET_FIELDS);
    for (const auto& group : encoding)
    {
        for (const auto& field : group.members)
        {
            packetFields[static_cast<size_t>(field.name)].reserve(numEvents);
        }
    }

    const uint8_t* packetByteStream = rawData;
    if (rawDataSize == 0) assert(numEvents == 0);

    auto decode = [](const SerializeParams& info, uint64_t val, auto& ptr, StoreSigned storeSigned)
    {
        return info.params.Visit(
            [&](const TruncateParams& info) { return TruncateOverflow::FromBinary(val, ptr, storeSigned, NumBits{info.numBits}); },
            [&](const CompactOverflowParams& info) { return CompactOverflow::FromBinary(val, ptr, storeSigned, NumBits{info.numBits}); },
            [&](const SimpleOverflowParams& info) { return SimpleOverflow::FromBinary(val, ptr, storeSigned, NumBits{info.numBits}, NumBytes{info.overflowBytes}); }
        );
    };

    while (packetByteStream < rawData + rawDataSize)
    {
        for (const auto& group : encoding)
        {
            uint64_t mainVal = 0;
            auto numBytes = (group.totalBits + 7) / 8;
            static_assert(CHAR_BIT == 8, "We assume 8 bits per byte.");
            std::memcpy(&mainVal, packetByteStream, numBytes);
            packetByteStream += numBytes;

            for (size_t i = 0; i < group.members.size(); ++i)
            {
                const auto& info = group.members[i];
                auto numBits = group.numBits[i];
                uint64_t val = mainVal & ((1 << numBits)-1);
                mainVal = mainVal >> numBits;
                auto integral = decode(info.serialize, val, packetByteStream, StoreSigned{info.storeSigned});
                packetFields[static_cast<size_t>(info.name)].push_back(integral);
            }
        }
    }

    return packetFields;
}

} // anon namespace

template<typename T>
void RawMetricData::ReadIntFromBaz(const MetricField& f, const uint8_t* buffer)
{
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) <= sizeof(int32_t),
                  "Can only read certain integer types from BAZ file!");
    T tmp;
    memcpy(&tmp, buffer, sizeof(T));
    idata_[whichVec_[f.fieldName]].emplace_back(static_cast<int32_t>(tmp));
}

template<typename T>
void RawMetricData::ReadUIntFromBaz(const MetricField& f, const uint8_t* buffer)
{
    static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value && sizeof(T) <= sizeof(uint32_t),
                  "Can only read certain unsigned integer types from BAZ file!");
    T tmp;
    memcpy(&tmp, buffer, sizeof(T));
    udata_[whichVec_[f.fieldName]].emplace_back(static_cast<uint32_t>(tmp));
}

template<typename T>
void RawMetricData::ReadFixedPointFromBaz(const MetricField& f, const uint8_t* buffer)
{
    static_assert(std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) <= sizeof(int16_t),
                  "Can ony read fixed point of signed integral types from BAZ file!");
    T tmp;
    memcpy(&tmp, buffer, sizeof(T));
    FloatFixedCodec<float,int16_t> codec(1/static_cast<float>(f.fieldScalingFactor));
    fdata_[whichVec_[f.fieldName]].emplace_back(codec.Decode(tmp));
}

void RawMetricData::ReadHalfFloatFromBaz(const MetricField& f, const uint8_t* buffer)
{
    half_float::half tmp;
    memcpy(&tmp, buffer, sizeof(tmp));
    fdata_[whichVec_[f.fieldName]].emplace_back(static_cast<float>(tmp));
}

size_t RawMetricData::ParseMetricField(const MetricField& f, const uint8_t* buffer)
{
    size_t bytesRead = 0;
    switch (f.fieldBitSize)
    {
        case 8:
            if (f.fieldSigned)
                ReadIntFromBaz<int8_t>(f, buffer);
            else
                ReadUIntFromBaz<uint8_t>(f, buffer);
            bytesRead = 1;
            break;
        case 16:
            if (f.fieldScalingFactor == 0)
                ReadHalfFloatFromBaz(f, buffer);
            else if (f.fieldScalingFactor == 1)
            {
                if (f.fieldSigned)
                    ReadIntFromBaz<int16_t>(f, buffer);
                else
                    ReadUIntFromBaz<uint16_t>(f, buffer);
            }
            else
            {
                ReadFixedPointFromBaz<int16_t>(f, buffer);
            }
            bytesRead = 2;
            break;
        case 32:
            // We don't allow storing single-precision floats.
            assert(f.fieldScalingFactor == 1);
            if (f.fieldSigned)
                ReadIntFromBaz<int32_t>(f, buffer);
            else
                ReadUIntFromBaz<uint32_t>(f, buffer);
            bytesRead = 4;
            break;
    }
    return bytesRead;
}

RawMetricData ParseMetricFields(const std::vector<MetricField>& fields, const ZmwByteData::ByteStream& data)
{
    //RawMetricData metrics(NUM_METRIC_FIELDS);
    RawMetricData metrics(fields);

    const uint8_t* buffer = data.get();
    size_t end = data.size();

    uint64_t bytesRead = 0;
    while (bytesRead < end)
    {
        for (const auto& f : fields)
        {
            if (bytesRead >= end)
                break;

            if (f.fieldName != MetricFieldName::GAP)
            {
                bytesRead += metrics.ParseMetricField(f, &buffer[bytesRead]);
            }
            else
            {
                bytesRead += 1 + f.fieldBitSize / 8;
            }
        }
    }

    return metrics;
}

BlockLevelMetrics ParseMetrics(const std::vector<MetricField>& metricFields,
                               uint32_t metricFrames,
                               double frameRateHz,
                               const std::vector<float> relAmps,
                               const std::string& baseMap,
                               const ZmwByteData& data, bool internal)
{
    if (!data.IsFull())
        throw PBException("Missing data when Parsing Metrics");

    // For now, all metrics must be at the same frequency
    RawMetricData rawMetrics;
    if (data.MetricByteStream().size() != 0)
    {
        rawMetrics = ParseMetricFields(metricFields, data.MetricByteStream());
    }
    else
    {
        PBLOG_WARN << "No metrics present to parse!";
    }
    return BlockLevelMetrics(rawMetrics, metricFrames, frameRateHz, relAmps, baseMap, internal);

    // TODO validate num_pulses and num_bases
}

RawEventData ParsePackets(const std::vector<GroupParams<PacketFieldName>>& packetGroups,
                          const std::vector<FieldParams<PacketFieldName>>& packetFields,
                          const ZmwByteData& data)
{

    if (!data.IsFull())
        throw PBException("Missing data when Parsing Metrics");

    auto rawData = ParsePacketFields(packetGroups, data.packetByteStream().get(), data.packetByteStream().size(), data.NumEvents());
    return RawEventData(std::move(rawData), packetFields);
}

RawEventData ParsePackets(const std::vector<BazIO::GroupParams<BazIO::PacketFieldName>>& groups,
                          const uint8_t* data,
                          size_t dataSize,
                          size_t numEvents)
{
    std::vector<FieldParams<BazIO::PacketFieldName>> params;
    for (const auto& g : groups)
    {
        for (const auto& f : g.members)
        {
            params.push_back(f);
        }
    }
    auto rawData = ParsePacketFields(groups, data, dataSize, numEvents);
    return RawEventData(std::move(rawData), params);
}

}} // PacBio::BazIO
