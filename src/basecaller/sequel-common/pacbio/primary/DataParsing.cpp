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

#include <pacbio/primary/DataParsing.h>

#include <cstring>
#include <half.hpp>

#include <pacbio/logging/Logger.h>
#include <pacbio/smrtdata/QvCore.h>
#include <pacbio/primary/PacketField.h>

using namespace PacBio::SmrtData::QvCore;

namespace PacBio {
namespace Primary {

namespace
{

const size_t NUM_PACKET_FIELDS = PacketFieldName::allValues().size();
static constexpr char tags[6] = {'A', 'C', 'G', 'T', 'N', '-'};

std::vector<std::vector<uint32_t>> ParsePacketFields(
        const std::vector<PacketField>& fields,
        const ZmwByteData::ByteStream& data,
        size_t numEvents)
{
    std::vector<std::vector<uint32_t>> packetFields(NUM_PACKET_FIELDS);
    for (size_t i = 0; i < NUM_PACKET_FIELDS; ++i)
        packetFields[i].reserve(numEvents);

    size_t counter = 0;
    uint64_t bytesRead = 0;
    uint64_t end = data.size();
    const uint8_t* packetByteStream = data.get();
    if (end == 0) assert(numEvents == 0);

    while (bytesRead < end)
    {
        for (const auto& f : fields)
        {
            if (bytesRead >= end) break;
            if (f.fieldName != PacketFieldName::GAP)
            {
                const auto fieldNameIndex = static_cast<uint8_t>(f.fieldName);
                if (f.fieldBitSize == 1)
                {
                    const uint8_t tmp8 = static_cast<uint8_t>((packetByteStream[bytesRead] & f.fieldBitMask)
                            >> f.fieldBitShift);
                    packetFields[fieldNameIndex].push_back(tmp8);
                    if (f.fieldBitShift == 0) ++bytesRead;
                }
                else if (f.fieldBitSize < 8)
                {
                    const uint8_t tmp8 = static_cast<uint8_t>((packetByteStream[bytesRead] & f.fieldBitMask)
                            >> f.fieldBitShift);

                    if (fieldNameIndex < 4) // READOUT, DEL_TAG, SUB_TAG, or LABEL
                    {
                        if (tmp8 > 4)
                            throw PBException("Woot is that tag? " + std::to_string(static_cast<int>(tmp8)));
                        packetFields[fieldNameIndex].emplace_back(tags[tmp8]);
                    }
                    else if (fieldNameIndex == 4) // ALT_LABEL
                    {
                        if (tmp8 > 5)
                            throw PBException("Woot is that tag? " + std::to_string(static_cast<int>(tmp8)));
                        packetFields[fieldNameIndex].emplace_back(tags[tmp8]);
                    }
                    else if (fieldNameIndex <= 10 || fieldNameIndex == 31) // QVs
                    {
                        packetFields[fieldNameIndex].emplace_back(tmp8 + 33);
                    }

                    if (f.fieldBitShift == 0) ++bytesRead;
                }
                else if (f.fieldBitSize == 8)
                {
                    if (!f.hasFieldEscape || packetByteStream[bytesRead] < f.fieldEscapeValue)
                    {
                        packetFields[fieldNameIndex].push_back(packetByteStream[bytesRead++]);
                    }
                    else if (f.extensionBitSize == 16)
                    {
                        uint16_t tmp16;
                        memcpy(&tmp16, &packetByteStream[++bytesRead],
                               sizeof(uint16_t)); // prefix increment to skip overflowed 8-bit escape value
                        packetFields[fieldNameIndex].push_back(tmp16);
                        bytesRead += 2;
                    }
                    else if (f.extensionBitSize == 32)
                    {
                        uint32_t tmp32;
                        memcpy(&tmp32, &packetByteStream[++bytesRead],
                               sizeof(uint32_t)); // prefix increment to skip overflowed 8-bit escape value
                        packetFields[fieldNameIndex].push_back(tmp32);
                        bytesRead += 4;
                    }
                    else
                    {
                        throw PBException(
                                "Didn't except that extension bit-size LL: " +
                                std::to_string(f.extensionBitSize) + "\t" +
                                f.fieldName.toString());
                    }
                }
                else
                {
                    throw PBException(
                            "Didn't except that field bit-size: " +
                            std::to_string(f.fieldBitSize));
                }
            }
            else
            {
                if (f.fieldBitSize < 8)
                {
                    if (f.fieldBitShift == 0)
                    {
                        ++bytesRead;
                    }
                }
                else if (f.fieldBitSize % 8 == 0)
                {
                    bytesRead += f.fieldBitSize / 8;
                }
                else
                {
                    throw PBException("Unknown/Gap field is cross byte. Not allowed.");
                }
            }
        }
        counter++;
    }

#if 0
    if (counter)
        std::cout << "packet " << bytesRead << " " << packetsByteSize << " " <<counter << " " << (double)bytesRead/counter << std::endl;
#endif

    // Compute OVERALL_QV
    uint8_t overallQvIndex = static_cast<uint8_t>(PacketFieldName::OVERALL_QV);
    if (packetFields[overallQvIndex].empty())
    {
        for (decltype(numEvents) i = 0; i < numEvents; ++i)
        {
            const auto pIns = errorProbability(
                    static_cast<int8_t>(packetFields[static_cast<uint8_t>(PacketFieldName::INS_QV)][i]));
            const auto pDel = errorProbability(
                    static_cast<int8_t>(packetFields[static_cast<uint8_t>(PacketFieldName::DEL_QV)][i]));
            const auto pSub = errorProbability(
                    static_cast<int8_t>(packetFields[static_cast<uint8_t>(PacketFieldName::SUB_QV)][i]));
            const auto q = (1 - pIns) * (1 - pDel) * (1 - pSub);
            const auto qv = qualityValue(1 - q) + 33;
            packetFields[overallQvIndex].push_back(qv);
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
    fdata_[whichVec_[f.fieldName]].emplace_back(static_cast<float>(tmp) / f.fieldScalingFactor);
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
                // Fixed-point scaling should have values stored as int16_t.
                assert(f.fieldSigned);
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

BlockLevelMetrics ParseMetrics(const FileHeader& fh, const ZmwByteData& data, bool internal)
{
    if (!data.IsFull())
        throw PBException("Missing data when Parsing Metrics");

    // For now, all metrics must be at the same frequency
    RawMetricData rawMetrics;
    MetricFrequency frequency;
    if (data.hFMByteStream().size() != 0)
    {
        if (data.mFMByteStream().size() != 0)
            throw PBException("Unexpected medium frequency metrics!");
        if (data.lFMByteStream().size() != 0)
            throw PBException("Unexpected low frequency metrics!");
        rawMetrics = ParseMetricFields(fh.HFMetricFields(), data.hFMByteStream());
        frequency = MetricFrequency::HIGH;
    }
    else if (data.mFMByteStream().size() != 0)
    {
        if (data.lFMByteStream().size() != 0)
            throw PBException("Unexpected low frequency metrics!");
        rawMetrics = ParseMetricFields(fh.MFMetricFields(), data.mFMByteStream());
        frequency = MetricFrequency::MEDIUM;
    }
    else if (data.lFMByteStream().size() > 0)
    {
        rawMetrics = ParseMetricFields(fh.LFMetricFields(), data.lFMByteStream());
        frequency = MetricFrequency::LOW;
    }
    else
    {
        PBLOG_WARN << "No metrics present to parse!";
        frequency = MetricFrequency::MEDIUM;
    }
    return BlockLevelMetrics(rawMetrics, fh, frequency, internal);

    // TODO validate num_pulses and num_bases
}

RawEventData ParsePackets(const FileHeader& fh, const ZmwByteData& data)
{

    if (!data.IsFull())
        throw PBException("Missing data when Parsing Metrics");

    auto rawData = ParsePacketFields(fh.PacketFields(), data.packetByteStream(), data.NumEvents());
    return RawEventData(std::move(rawData));
}

}}
