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
#include <utility>
#include <tuple>
#include <cstring>

#include <bazio/MetricBlock.h>
#include <bazio/BazCore.h>
#include <bazio/BlockActivityLabels.h>
#include <bazio/Timing.h>

#include "SimulateWriteUtils.h"

#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

using PacBio::SmrtData::NucleotideLabel;

namespace PacBio::Primary::Postprimary
{

/// Produces random numbers for given ranges
class SimulationRNG
{
public: // structors
    SimulationRNG()
    {
        std::uniform_int_distribution<uint8_t> oneBitRng(0, 2);
        std::uniform_int_distribution<uint8_t> twoBitRng(0, 3);
        std::uniform_int_distribution<uint8_t> threeBitRng(0, 4);
        std::uniform_int_distribution<uint8_t> fourBitRng(0, 15);
        std::uniform_int_distribution<uint8_t> eightBitRng(0, 254);
        std::uniform_int_distribution<uint16_t> twoByteRng(0, 60000);
        std::uniform_int_distribution<uint32_t> fourByteRng(0, 100000);

        oneBitRng_.reserve(maxRng_);
        twoBitRng_.reserve(maxRng_);
        fourBitRng_.reserve(maxRng_);
        threeBitRng_.reserve(maxRng_);
        eightBitRng_.reserve(maxRng_);
        twoByteRng_.reserve(maxRng_);
        fourByteRng_.reserve(maxRng_);

        for (int i = 0; i < maxRng_; ++i)
        {
            oneBitRng_.push_back(oneBitRng(generator_));
            twoBitRng_.push_back(twoBitRng(generator_));
            threeBitRng_.push_back(threeBitRng(generator_));
            fourBitRng_.push_back(fourBitRng(generator_));
            eightBitRng_.push_back(eightBitRng(generator_));
            twoByteRng_.push_back(twoByteRng(generator_));
            fourByteRng_.push_back(fourByteRng(generator_));
        }
    }
    // Move constructor
    SimulationRNG(SimulationRNG&&) = delete;
    // Copy constructor
    SimulationRNG(const SimulationRNG&) = delete;
    // Move assignment operator
    SimulationRNG& operator=(SimulationRNG&& rhs) noexcept = delete;
    // Copy assignment operator
    SimulationRNG& operator=(const SimulationRNG&) = delete;

public:
    std::vector<SpiderMetricBlock> SimulateMetrics(const uint16_t numMetricBlocks)
    {
        std::vector<SpiderMetricBlock> metrics;
        metrics.resize(numMetricBlocks);
        for (int i = 0; i < numMetricBlocks; ++i)
        {
            const auto basesPerAnalog = NextTwoByte();
            const auto pulses = 4 * basesPerAnalog + NextEightBit();
            const auto pulseMetric = pulses == 0 ? pulses : pulses - 1;
            const float bl = NextFourBit();
            metrics[i].NumBasesA(basesPerAnalog)
                      .NumBasesC(basesPerAnalog)
                      .NumBasesG(basesPerAnalog)
                      .NumBasesT(basesPerAnalog)
                      .BaselineSds({{bl}})
                      .NumPulses(pulses)
                      .PulseWidth(NextTwoByte())
                      .BaseWidth(NextTwoByte())
                      .Baselines({{1}})
                      .PkmidA(bl * 10)
                      .PkmidC(bl * 10)
                      .PkmidG(bl * 10)
                      .PkmidT(bl * 10)
                      .NumPkmidFramesA(NextEightBit())
                      .NumPkmidFramesC(NextEightBit())
                      .NumPkmidFramesG(NextEightBit())
                      .NumPkmidFramesT(NextEightBit())
                      .NumFrames(NextEightBit())
                      .NumBaselineFrames({{NextEightBit()}})
                      .NumSandwiches(pulseMetric)
                      .NumHalfSandwiches(pulseMetric)
                      .NumPulseLabelStutters(pulseMetric)
                      .NumFrames(1024)
                      .ActivityLabel(ActivityLabeler::A1);
        }
        return metrics;
    }
    std::unique_ptr<SimPulse[]> SimulateBaseCalls(
            const uint16_t numEvents,
            uint64_t* currentPulseFrame,
            uint64_t* currentBaseFrame,
            bool internal = false)
    {
        auto basecall = std::make_unique<SimPulse[]>(numEvents);
        for (int i = 0; i < numEvents; ++i)
        {
            bool isBase;
            int ipd = NextFourBit() >= 14 ? NextFourByte() : NextEightBit();
            if (internal)
            {
                isBase = NextOneBit() != 0;
                basecall[i].Label(isBase ? NextBase() : SimPulse::NucleotideLabel::A);
            }
            else
            {
                isBase = true;
                basecall[i].Label(NextBase());
            }
            basecall[i].IsReject(!isBase);
            basecall[i].MeanSignal(NextTwoByte()/10.0);
            basecall[i].MidSignal(NextTwoByte()/10.0);
            basecall[i].Start(*currentPulseFrame + ipd);
            basecall[i].Width(NextFourBit() >= 14 ? NextTwoByte() : NextEightBit());
            *currentPulseFrame += ipd + basecall[i].Width();
            if (isBase)
            {
                *currentBaseFrame = *currentPulseFrame;
            }
        }
        return basecall;
    }
    std::unique_ptr<SimPulse[]> SimulateBaseCalls(
            const std::string& sequence, size_t* numEvents,
            uint64_t* currentPulseFrame,
            uint64_t* currentBaseFrame,
            bool internal = false)
    {
        std::vector<SimPulse> basecall;
        basecall.reserve(sequence.size() * 3);
        size_t i = 0;
        while (i < sequence.size())
        {
            SimPulse b;
            bool isBase;
            int ipd = NextFourBit() >= 14 ? NextFourByte() : NextEightBit();
            uint8_t cbase = sequence.at(i);
            SimPulse::NucleotideLabel base;
            switch(cbase)
            {
                case 'A': base = SimPulse::NucleotideLabel::A; break;
                case 'C': base = SimPulse::NucleotideLabel::C; break;
                case 'G': base = SimPulse::NucleotideLabel::G; break;
                case 'T': base = SimPulse::NucleotideLabel::T; break;
                default: throw std::runtime_error("Not aware of base " + std::to_string(cbase));
            }
            if (internal)
            {
                isBase = NextOneBit() != 0;
                b.Label(isBase ? base : SimPulse::NucleotideLabel::NONE);
            }
            else
            {
                isBase = true;
                b.Label(base);
            }
            b.IsReject(!isBase);
            b.MeanSignal(NextTwoByte()/10.0);
            b.MidSignal(NextTwoByte()/10.0);
            b.Start(*currentPulseFrame + ipd);
            b.Width(std::max(NextFourBit() >= 14 ? NextTwoByte() : NextEightBit(), 1));
            *currentPulseFrame += ipd + b.Width();
            basecall.emplace_back(std::move(b));
            if (isBase)
            {
                ++i;
                *currentBaseFrame = *currentPulseFrame;
            }
        }
        (*numEvents) = basecall.size();
        auto basecallArray = std::make_unique<SimPulse[]>(basecall.size());
        std::copy(basecall.begin(), basecall.end(), basecallArray.get());
        return basecallArray;
    }

    inline SimPulse::NucleotideLabel NextBase()
    { return ToNucleotideLabel(NextTwoBit()); }

    inline SimPulse::NucleotideLabel ToNucleotideLabel(uint8_t value)
    {
        switch(value)
        {
            case 0: return SimPulse::NucleotideLabel::A;
            case 1: return SimPulse::NucleotideLabel::C;
            case 2: return SimPulse::NucleotideLabel::G;
            case 3: return SimPulse::NucleotideLabel::T;
            default:
                throw std::runtime_error("Not aware of that base in simulation");
        }
    }

    inline uint8_t NextOneBit()
    {
        if (oneBitCounter_ == maxRng_) oneBitCounter_ = 0;
        return oneBitRng_[oneBitCounter_++];
    }

    inline uint8_t NextTwoBit()
    {
        if (twoBitCounter_ == maxRng_) twoBitCounter_ = 0;
        return twoBitRng_[twoBitCounter_++];
    }

    inline uint8_t NextThreeBit()
    {
        if (threeBitCounter_ == maxRng_) threeBitCounter_ = 0;
        return threeBitRng_[threeBitCounter_++];
    }

    inline uint8_t NextFourBit()
    {
        if (fourBitCounter_ == maxRng_) fourBitCounter_ = 0;
        return fourBitRng_[fourBitCounter_++];
    }

    inline uint8_t NextEightBit()
    {
        if (eightBitCounter_ == maxRng_) eightBitCounter_ = 0;
        return eightBitRng_[eightBitCounter_++];
    }

    inline uint16_t NextTwoByte()
    {
        if (twoByteCounter_ == maxRng_) twoByteCounter_ = 0;
        return twoByteRng_[twoByteCounter_++];
    }

    inline uint32_t NextFourByte()
    {
        if (fourByteCounter_ == maxRng_) fourByteCounter_ = 0;
        return fourByteRng_[fourByteCounter_++];
    }

private: // data
    const int maxRng_ = 1000000;

    std::mt19937 generator_;

    std::vector<uint8_t> oneBitRng_;
    std::vector<uint8_t> twoBitRng_;
    std::vector<uint8_t> threeBitRng_;
    std::vector<uint8_t> fourBitRng_;
    std::vector<uint8_t> eightBitRng_;
    std::vector<uint16_t> twoByteRng_;
    std::vector<uint32_t> fourByteRng_;

    int oneBitCounter_ = 0;
    int twoBitCounter_ = 0;
    int threeBitCounter_ = 0;
    int fourBitCounter_ = 0;
    int eightBitCounter_ = 0;
    int twoByteCounter_ = 0;
    int fourByteCounter_ = 0;
};

}

