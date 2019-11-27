// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
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

// Programmer: Armin Töpfer

#pragma once

#define BYTETOBINARY(byte)  \
  (byte & 0x80 ? 1 : 0), \
  (byte & 0x40 ? 1 : 0), \
  (byte & 0x20 ? 1 : 0), \
  (byte & 0x10 ? 1 : 0), \
  (byte & 0x08 ? 1 : 0), \
  (byte & 0x04 ? 1 : 0), \
  (byte & 0x02 ? 1 : 0), \
  (byte & 0x01 ? 1 : 0) 

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <cstring>

#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/SimulationRNG.h>

namespace PacBio {
namespace Primary {

/// Simulates an internal mode BAZ
class SimulationFuture
{
public: // structors
    SimulationFuture(const std::string& fileName,
                     const std::string& chipLayoutName,
                     const std::vector<uint32_t>& zmwNumbers,
                     int zmws, double bps,
                     int seconds, int chunks, bool silent)
        : fileName_(fileName)
        , chipLayoutName_(chipLayoutName)
        , zmwNumbers_(zmwNumbers)
        , zmws_(zmws)
        , bps_(bps)
        , seconds_(seconds)
        , chunks_(chunks)
        , silent_(silent)
    {
        Simulate();
    }
    // Default constructor
    SimulationFuture() = delete;
    // Move constructor
    SimulationFuture(SimulationFuture&&) = delete;
    // Copy constructor
    SimulationFuture(const SimulationFuture&) = delete;
    // Move assignment operator
    SimulationFuture& operator=(SimulationFuture&& rhs) noexcept = delete;
    // Copy assignment operator
    SimulationFuture& operator=(const SimulationFuture&) = delete;

private: // data
    std::string fileName_;
    std::string chipLayoutName_;
    std::vector<uint32_t> zmwNumbers_;
    int zmws_;
    double bps_;
    int seconds_;
    int chunks_;
    bool silent_;

private:
    void Simulate()
    {
        (void) fileName_;
        (void) chipLayoutName_;
        (void) zmwNumbers_;
        (void) zmws_;
        (void) bps_;
        (void) seconds_;
        (void) chunks_;
        (void) silent_;
        // std::stringstream ss;
        // ss << "{\"TYPE\":\"BAZ\", \"HEADER\":"
        //    << "{\"MOVIE_NAME\":\"ArminsFakeMovie\",\"BASE_CALLER_VERSION\":\"1."
        //       "2\",\"BAZWRITER_VERSION\":\""
        //       << BAZIO_VERSION <<"\","
        //       "\"BAZ_MAJOR_VERSION\":0,\"BAZ_MINOR_VERSION\":1,\"BAZ_PATCH_"
        //       "VERSION\":0,\"FRAME_RATE_HZ\":100,\"HF_METRIC\":{\"FIELDS\":[["
        //       "\"PR_0\",8],[\"PR_1\",8],[\"NUM_PULSES\",16]],\"FRAMES\":512},"
        //       "\"LF_METRIC\":{\"FIELDS\":[[\"BASELINE_LEVEL_RED\",8],[\"PULSE_"
        //       "WIDTH\",16],[\"BASE_WIDTH\",16],[\"BASELINE_LEVEL_GREEN\",8],["
        //       "\"BASELINE_STANDARD_STD\",16],[\"NUM_PULSES\",16],[\"NUM_"
        //       "BASES\","
        //       "16]],\"FRAMES\":16384},\"MF_METRIC\":{\"FIELDS\":[[\"PKMID_A\","
        //       "8],"
        //       "[\"PKMID_C\",8],[\"PKMID_G\",8],[\"PKMID_T\",8],[\"NUM_PULSES\","
        //       "16],[\"NUM_BASES\",16]],\"FRAMES\":4096},\"PACKET\":[[\"BASE_"
        //       "CALL\",2],[\"DEL_TAG\",3],[\"SUB_TAG\",3],[\"DEL_QV\",4],[\"SUB_"
        //       "QV\",4],[\"INS_QV\",4],[\"MRG_QV\",4],[\"ALT_QV\",4],[\"LAB_"
        //       "QV\",4],[\"ALT_LABEL\",4],[\"LABEL\",4],[\"FUTURE\",8],[\"IPD_"
        //       "LL\",8,255,\"IPD16_"
        //       "LL\",16],[\"PW_LL\",8,255,\"PW16_LL\",16],[\"PKMEAN_LL\",8,255,"
        //       "\"PKMEAN16_LL\",16],[\"PKMID_LL\",8,255,\"PKMID16_LL\",16]],"
        //       "\"SLICE_LENGTH_FRAMES\":16384}}";
        // std::string json = ss.str();

        // std::vector<char> header(json.begin(), json.end());

        // BazWriter writer(fileName_, header, BazMode::INTERNAL, silent_);
        // for (int c = 0; c < chunks_; ++c)
        // {
        //     auto now = std::chrono::high_resolution_clock::now();
        //     for (int i = 0; i < zmws_; ++i)
        //     {
        //         auto numEvents = bps_ * seconds_;
        //         uint16_t packetsByteSize = 10 * numEvents + 4 * 2 * ((numEvents - 1) / 20 + 1);
        //         auto bases = randomBases(numEvents, packetsByteSize);
        //         auto metrics = randomMetrics(32);
        //         auto zs = std::unique_ptr<ZmwSlice>(new ZmwSlice(std::move(bases), std::move(metrics)));
        //         zs->NumEvents(numEvents)
        //             .PacketsByteSize(packetsByteSize)
        //             .NumHFMBs(32)
        //             .ZmwNum(i);
        //         writer.AddZmwSlice(std::move(zs));
        //     }
        //     Timing::PrintTime(now, "SimulationFuture");
        //     writer.Flush();
        // }
    }

    SmrtBytePtr randomBases(int numEvents, uint16_t& packetsByteSize)
    {
        auto bases = SmartMemory::AllocBytePtr(packetsByteSize);

        size_t offset = 0;
        for (int i = 0; i < numEvents; ++i)
        {
            uint16_t ipd_16 = static_cast<uint16_t>(!(i % 20) ? 65535 : 8);
            offset += AppendProductionBaseCallToBuffer(bases.get() + offset,
                                                       0,
                                                       1,
                                                       2,
                                                       3,
                                                       4,
                                                       5,
                                                       6,
                                                       7,
                                                       8,
                                                       1,
                                                       2,
                                                       ipd_16);
        }
        if (offset != packetsByteSize) 
            throw std::runtime_error(":( " + std::to_string(offset) + " " + std::to_string(packetsByteSize));
        return bases;
    }

    uint64_t AppendProductionBaseCallToBuffer(uint8_t* buffer, uint8_t baseCall, uint8_t delTag,
                            uint8_t subTag, uint8_t delQV, uint8_t mrgQV, uint8_t subQV,
                            uint8_t insQV, uint8_t altQV, uint8_t labQV, 
                            uint8_t label, uint8_t altLabel,
                            uint16_t ipd) 
    {
        assert(baseCall < 5);
        assert(delTag < 5);
        assert(subTag < 5);
        uint64_t counter = 0;
        buffer[counter++] = static_cast<uint8_t>((baseCall << 3 | delTag) << 3 | subTag); // BaseCall, DelTag, SubTag  1
        buffer[counter++] = static_cast<uint8_t>(delQV << 4 | mrgQV);                     // DelQV, MrgQV              2
        buffer[counter++] = static_cast<uint8_t>(subQV << 4 | insQV);                     // SubQV, InsQV              3
        buffer[counter++] = static_cast<uint8_t>(altQV << 4 | labQV);                     // ClaQV, LabQV              4
        buffer[counter++] = static_cast<uint8_t>(altLabel << 4 | label);                  // AltLabel, Label           5
        buffer[counter++] = 42;                                     // FUTURE                    6

        if (ipd > 255) // 16 bit IPD                                                             7
        {
            buffer[counter++] = 255;
            memcpy(buffer+counter, &ipd, sizeof(uint16_t));
            counter += sizeof(uint16_t);
        }
        else // 8 bit IPD
        {
            buffer[counter++] = static_cast<uint8_t>(ipd);
        }
        if (ipd > 255) // 16 bit IPD                                                             8
        {
            buffer[counter++] = 255;
            memcpy(buffer+counter, &ipd, sizeof(uint16_t));
            counter += sizeof(uint16_t);
        }
        else // 8 bit IPD
        {
            buffer[counter++] = static_cast<uint8_t>(ipd);
        }
        if (ipd > 255) // 16 bit IPD                                                             9
        {
            buffer[counter++] = 255;
            memcpy(buffer+counter, &ipd, sizeof(uint16_t));
            counter += sizeof(uint16_t);
        }
        else // 8 bit IPD
        {
            buffer[counter++] = static_cast<uint8_t>(ipd);
        }
        if (ipd > 255) // 16 bit IPD                                                            10
        {
            buffer[counter++] = 255;
            memcpy(buffer+counter, &ipd, sizeof(uint16_t));
            counter += sizeof(uint16_t);
        }
        else // 8 bit IPD
        {
            buffer[counter++] = static_cast<uint8_t>(ipd);
        }
        return counter;
    }

    SmrtMemPtr<SequelMetricBlock> randomMetrics(int numMetrics)
    {
        auto metricBlocks = SmartMemory::AllocMemPtr<SequelMetricBlock>(numMetrics);

        for (int i = 0; i < numMetrics; ++i)
        {
            metricBlocks.get()[i].BaselineSds({2, 1})
                           .NumPulses(3)
                           .PulseWidth(4)
                           .BaseWidth(5)
                           .Baselines({9, 8})
                           .PkmidA(10)
                           .PkmidC(11)
                           .PkmidG(12)
                           .PkmidT(13);
        }
        return metricBlocks;
    }
};

}}


