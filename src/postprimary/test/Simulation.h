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

#include <pacbio/datasource/ZmwFeatures.h>
#define BYTETOBINARY(byte)  \
  (byte & 0x80 ? 1 : 0), \
  (byte & 0x40 ? 1 : 0), \
  (byte & 0x20 ? 1 : 0), \
  (byte & 0x10 ? 1 : 0), \
  (byte & 0x08 ? 1 : 0), \
  (byte & 0x04 ? 1 : 0), \
  (byte & 0x02 ? 1 : 0), \
  (byte & 0x01 ? 1 : 0)

#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <bazio/BazCore.h>
#include <bazio/Timing.h>
#include <bazio/encoding/test/TestingPulse.h>

#include "SimulateConfigs.h"
#include "SimulationRNG.h"
#include "SimulateWriteUtils.h"

#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

using namespace PacBio::SmrtData;

namespace PacBio::Primary::Postprimary
{

/// Simulates a BASES readout BAZ
class Simulation
{
public:
    static BazIO::ZmwInfo SimulateZmwInfo(const std::vector<uint32_t>& zmwNumbers)
    {
        // Simulate out matching XY-coordinate to match the hole numbers,
        // hole types all set to Sequencing and the unit features to be 0.
        std::vector<uint16_t> zmwX;
        std::vector<uint16_t> zmwY;
        for (size_t i = 0; i < zmwNumbers.size(); i++)
        {
            zmwX.push_back((zmwNumbers[i] & 0xFFFF0000) >> 16);
            zmwY.push_back(zmwNumbers[i] & 0x0000FFFF);
        }
        std::iota(zmwX.begin(), zmwX.end(), 0);
        BazIO::ZmwInfo zmwInfo(BazIO::ZmwInfo::Data {
                                   zmwNumbers,
                                   std::vector<uint8_t>(zmwNumbers.size(), 1),
                                   zmwX,
                                   zmwY,
                                   std::vector<uint32_t>(zmwNumbers.size(), DataSource::ZmwFeatures::Sequencing)
                               });
        return zmwInfo;
    }

public: // structors
    Simulation(const std::string& fileName,
               const std::vector<uint32_t>& zmwNumbers,
               const int zmws, const double bps,
               const int seconds, const int chunks, bool silent,
               bool realtimeActivityLabels=false,
               const std::vector<float>& relativeAmplitudes=std::vector<float>{1, 0.946, 0.529, 0.553})
        : fileName_(fileName)
        , zmwNumbers_(zmwNumbers)
        , zmws_(zmws)
        , bps_(bps)
        , seconds_(seconds)
        , chunks_(chunks)
        , silent_(silent)
        , baseMap_("CTAG")
        , realtimeActivityLabels_(realtimeActivityLabels)
        , relativeAmplitudes_(relativeAmplitudes)
    {}
    // Default constructor
    Simulation() = delete;
    // Move constructor
    Simulation(Simulation&&) = delete;
    // Copy constructor
    Simulation(const Simulation&) = delete;
    // Move assignment operator
    Simulation& operator=(Simulation&& rhs) noexcept = delete;
    // Copy assignment operator
    Simulation& operator=(const Simulation&) = delete;

private: // data
    SimulationRNG rng;
    std::string fileName_;
    std::vector<uint32_t> zmwNumbers_;
    int zmws_;
    double bps_;
    int seconds_;
    int chunks_;
    bool silent_;
    std::string baseMap_;
    bool realtimeActivityLabels_;
    std::vector<float> relativeAmplitudes_;
    bool summarize_;
    const uint32_t metricFrames = 1024;
    const uint32_t superChunkFrames = 16384;
    const uint16_t numMetrics = superChunkFrames / metricFrames;

public:

    BazIO::FileHeaderBuilder GetFileHeaderBuilder(const Readout readout)
    {
        std::vector<uint32_t> emptyList;
        double frameRate = 100.0;
        using FileHeaderBuilder = BazIO::FileHeaderBuilder;
        FileHeaderBuilder fhb("m00001_052415_013000",
                              frameRate,
                              frameRate * chunks_ * seconds_,
                              (readout != Readout::PULSES)
                              ? BazIO::ProductionPulses::Params() : BazIO::InternalPulses::Params(),
                              generateExperimentMetadata(),
                              generateBasecallerConfig(),
                              SimulateZmwInfo(zmwNumbers_),
                              metricFrames,
                              FileHeaderBuilder::Flags()
                                .RealTimeActivityLabels(realtimeActivityLabels_)
        );
        return fhb;
    }
    void SimulateTrivial()
    {
        const Readout readout = Readout::BASES_WITHOUT_QVS;
        const auto numEvents = static_cast<uint32_t>(bps_ * seconds_);
        auto basecall = std::make_unique<SimPulse[]>(numEvents);
        int currentFrame = 0;
        for (uint32_t i = 0; i < numEvents; ++i)
        {
            int ipd = 20;
            basecall[i].Label(SimPulse::NucleotideLabel::A);
            basecall[i].MeanSignal(20/10.0);
            basecall[i].MidSignal(20/10.0);
            basecall[i].Start(currentFrame + ipd);
            basecall[i].Width(20);
            currentFrame += ipd + basecall[i].Width();
        }

        BazIO::FileHeaderBuilder fhb = GetFileHeaderBuilder(readout);

        SimBazWriter writer(fileName_, fhb, PacBio::Primary::BazIOConfig{}, silent_);

        for (int c = 0; c < chunks_; ++c)
        {
            // Random for shuffling zmwIds
            std::random_device rd;
            std::mt19937 g(rd());

            std::vector<int> zmwIds;
            zmwIds.reserve(zmws_);
            for (int i = 0; i < zmws_; ++i)
                zmwIds.emplace_back(i);

            std::shuffle(zmwIds.begin(), zmwIds.end(), g);

            // Simulate
            const auto now = std::chrono::high_resolution_clock::now();
            for (const auto i : zmwIds)
            {
                std::vector<SpiderMetricBlock> hfMetric;
                hfMetric.resize(numMetrics);
                for (int ii = 0; ii < numMetrics; ++ii)
                {
                    hfMetric[ii].NumFrames(1024)
                                .NumPulses(20)
                                .PulseWidth(20)
                                .BaseWidth(20)
                                .NumSandwiches(10)
                                .NumHalfSandwiches(5)
                                .NumPulseLabelStutters(12)
                                .PulseDetectionScore(0.5f)
                                .TraceAutocorr(0.3f)
                                .PixelChecksum(21)
                                .DmeStatus(3)
                                .ActivityLabel(ActivityLabeler::A1)
                                .PkmidA(8)
                                .PkmidC(8)
                                .PkmidG(8)
                                .PkmidT(8)
                                .NumPkmidFramesA(12)
                                .NumPkmidFramesC(12)
                                .NumPkmidFramesG(12)
                                .NumPkmidFramesT(12)
                                .PkmaxA(12.3f)
                                .PkmaxC(2.8f)
                                .PkmaxG(8.4f)
                                .PkmaxT(7.9f)
                                .NumBasesA(5)
                                .NumBasesC(5)
                                .NumBasesG(5)
                                .NumBasesT(5)
                                .NumPkmidBasesA(3)
                                .NumPkmidBasesC(3)
                                .NumPkmidBasesG(3)
                                .NumPkmidBasesT(3)
                                .BpzvarA(3.4f)
                                .BpzvarC(7.3f)
                                .BpzvarG(4.2f)
                                .BpzvarT(9.8f)
                                .PkzvarA(3.4f)
                                .PkzvarC(7.3f)
                                .PkzvarG(4.2f)
                                .PkzvarT(9.8f)
                                .Baselines({8})
                                .BaselineSds({20})
                                .NumBaselineFrames({10});
                }
                writer.AddZmwSlice(basecall.get(), numEvents, std::move(hfMetric), i);
            }
            if (!silent_) Timing::PrintTime(now, "Simulation");

            writer.Flush();
        }
        writer.WaitForTermination();
    }

    void Simulate(const Readout readout, uint16_t numMetrics)
    {
        BazIO::FileHeaderBuilder fhb = GetFileHeaderBuilder(readout);

        if (numMetrics == 0) fhb.ClearMetricFields();
        SimBazWriter writer(fileName_, fhb, PacBio::Primary::BazIOConfig{}, silent_);

        std::vector<uint64_t> currentPulseFrames(zmws_, 0);
        std::vector<uint64_t> currentBaseFrames(zmws_, 0);

        for (int c = 0; c < chunks_; ++c)
        {
            // Random for shuffling zmwIds
            std::random_device rd;
            std::mt19937 g(rd());

            std::vector<int> zmwIds;
            zmwIds.reserve(zmws_);
            for (int i = 0; i < zmws_; ++i)
                zmwIds.emplace_back(i);

            std::shuffle(zmwIds.begin(), zmwIds.end(), g);;

            // Simulate
            const auto now = std::chrono::high_resolution_clock::now();
            for (const auto i : zmwIds)
            {
                const auto numEvents = static_cast<uint32_t>(bps_ * seconds_);
                auto basecall = rng.SimulateBaseCalls(numEvents, &currentPulseFrames[i], &currentBaseFrames[i], readout == Readout::PULSES);
                std::vector<SpiderMetricBlock> metric = rng.SimulateMetrics(numMetrics);
                writer.AddZmwSlice(basecall.get(), numEvents, std::move(metric), i);

            }
            if (!silent_) Timing::PrintTime(now, "Simulation");
            writer.Flush();
        }

        if (summarize_)
        {
            writer.Summarize(std::cout);
        }
    }

    void SimulateBases()
    {
        Simulate(Readout::BASES, numMetrics);
    }

    void SimulatePulses()
    {
        Simulate(Readout::PULSES, numMetrics);
    }

    void SimulateBasesNoMetrics()
    {
        Simulate(Readout::BASES, 0);
    }

    void SimulatePulsesNoMetrics()
    {
        Simulate(Readout::PULSES, 0);
    }

    void Summarize(bool flag)
    {
        summarize_ = flag;
    }
};

}

