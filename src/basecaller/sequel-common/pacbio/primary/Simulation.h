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

#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/SimulationRNG.h>

#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

using namespace PacBio::SmrtData;

namespace PacBio {
namespace Primary {

std::string generateExperimentMetadata(const std::string& chipLayoutName="",
                                       const std::vector<float>& relamps=std::vector<float>{1, 0.946, 0.529, 0.553},
                                       const std::string& basemap="CTAG");

std::string generateBasecallerConfig(const std::string& pipename);

/// Simulates a BASES readout BAZ
class Simulation
{
public: // structors
    Simulation(const std::string& fileName,
               const std::string& chipLayoutName,
               const std::vector<uint32_t>& zmwNumbers,
               const int zmws, const double bps,
               const int seconds, const int chunks, bool silent,
               const std::vector<float>& relativeAmplitudes=std::vector<float>{1, 0.946, 0.529, 0.553})
        : fileName_(fileName)
        , chipLayoutName_(chipLayoutName)
        , zmwNumbers_(zmwNumbers)
        , zmws_(zmws)
        , bps_(bps)
        , seconds_(seconds)
        , chunks_(chunks)
        , silent_(silent)
        , baseMap_("CTAG")
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
    std::string chipLayoutName_;
    std::vector<uint32_t> zmwNumbers_;
    int zmws_;
    double bps_;
    int seconds_;
    int chunks_;
    bool silent_;
    std::string baseMap_;
    std::vector<float> relativeAmplitudes_;
    bool summarize_;
public:

    FileHeaderBuilder GetFileHeaderBuilder(const Readout readout, const MetricsVerbosity verbosity)
    {
        std::vector<uint32_t> emptyList;
        double frameRate = 100.0;
        FileHeaderBuilder fhb("m00001_052415_013000",
                              frameRate,
                              frameRate * chunks_ * seconds_,
                              readout,
                              verbosity,
                              generateExperimentMetadata(chipLayoutName_, relativeAmplitudes_, baseMap_),
                              generateBasecallerConfig(chipLayoutName_.find("Sequ") == 0 ? "Sequel" : "Spider"),
                              zmwNumbers_,
                              emptyList,
                              1024, // hFMetricFrames
                              4096, // mFMetricFrames
                              16384, // sliceLengthFrames
                              false,  // spiderOnSequel
                              true, // newBazFormat
                              true, // half-float
                              false  // realtimeActivityLabels
        );
        return fhb;
    }
    void SimulateTrivial()
    {
        const Readout readout = Readout::BASES_WITHOUT_QVS;
        const MetricsVerbosity verbosity = MetricsVerbosity::MINIMAL;
        const uint16_t numMetrics = 16;
        const auto numEvents = static_cast<uint32_t>(bps_ * seconds_);
        Basecall* basecall = new Basecall[numEvents];
        int currentFrame = 0;
        for (uint32_t i = 0; i < numEvents; ++i)
        {
            int ipd = 20;
            basecall[i].Base(NucleotideLabel::A);
            basecall[i].DeletionTag(NucleotideLabel::C);
            basecall[i].SubstitutionTag(NucleotideLabel::C);
            basecall[i].DeletionQV(5);
            basecall[i].GetPulse().MergeQV(5);
            basecall[i].SubstitutionQV(5);
            basecall[i].InsertionQV(5);
            basecall[i].GetPulse().AltLabelQV(5);
            basecall[i].GetPulse().LabelQV(5);
            basecall[i].GetPulse().AltLabel(NucleotideLabel::C);
            basecall[i].GetPulse().Label(NucleotideLabel::C);
            basecall[i].GetPulse().MeanSignal(20/10.0);
            basecall[i].GetPulse().MidSignal(20/10.0);
            basecall[i].GetPulse().Start(currentFrame + ipd);
            basecall[i].GetPulse().Width(20);
            currentFrame += ipd + basecall[i].GetPulse().Width();
        }

        FileHeaderBuilder fhb = GetFileHeaderBuilder(readout, verbosity);

        BazWriter<SequelMetricBlock> writer(fileName_, fhb, PacBio::Primary::BazIOConfig{}, silent_);

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
            bool success;
            for (const auto i : zmwIds)
            {
                std::vector<SequelMetricBlock> hfMetric;
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
                                .ActivityLabel(ActivityLabeler::HQRFPhysicalState::SINGLE)
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
                                .Baselines({8,8})
                                .BaselineSds({20,20})
                                .NumBaselineFrames({10,10})
                                .Angles({28.3f,72.5f});
                }
                success = writer.AddZmwSlice(basecall, numEvents, std::move(hfMetric), i);
                if (!success)
                {
                    std::cerr << "Problem writing BAZ: " << writer.ErrorMessage() << std::endl;
                    return;
                }
            }
            if (!silent_) Timing::PrintTime(now, "Simulation");

            success = writer.Flush();
            if (!success)
            {
                std::cerr << "Problem writing BAZ: " << writer.ErrorMessage() << std::endl;
                return;
            }

            // Avoid flooding
            // while (writer.BufferSize() > 1)
            // {
            //     if (!silent_) std::cerr << "WAIT" << std::endl;
            //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
            // }
        }
        delete[] basecall;
    }

    void Simulate(const Readout readout, const MetricsVerbosity verbosity, uint16_t numMetrics)
    {
        FileHeaderBuilder fhb = GetFileHeaderBuilder(readout, verbosity);

        if (numMetrics == 0) fhb.ClearAllMetricFields();
        BazWriter<SequelMetricBlock> writer(fileName_, fhb, PacBio::Primary::BazIOConfig{}, silent_);

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
            bool success;
            for (const auto i : zmwIds)
            {
                const auto numEvents = static_cast<uint32_t>(bps_ * seconds_);
                Basecall* basecall = rng.SimulateBaseCalls(numEvents, &currentPulseFrames[i], &currentBaseFrames[i], readout == Readout::PULSES);
                std::vector<SequelMetricBlock> hfMetric = rng.SimulateHFMetrics(numMetrics);
                success = writer.AddZmwSlice(basecall, numEvents, std::move(hfMetric), i);
                delete[] basecall;

                if (!success)
                {
                    std::cerr << "Problem writing BAZ: " << writer.ErrorMessage() << std::endl;
                    return;
                }
            }
            if (!silent_) Timing::PrintTime(now, "Simulation");
            success = writer.Flush();
            if (!success)
            {
                std::cerr << "Problem writing BAZ: " << writer.ErrorMessage() << std::endl;
                return;
            }

            // Avoid flooding
            while (writer.BufferSize() > 1)
            {
                if (!silent_) std::cerr << "WAIT" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        writer.WaitForTermination();

        if (summarize_)
        {
            fhb.EstimatesSummary(std::cout, bps_);
            writer.Summarize(std::cout);
        }
    }

    void SimulateBases() 
    {
        Simulate(Readout::BASES, MetricsVerbosity::MINIMAL, 16);
    }

    void SimulatePulses() 
    {
        Simulate(Readout::PULSES, MetricsVerbosity::HIGH, 16);
    }

    void SimulateBasesNoMetrics() 
    {
        Simulate(Readout::BASES, MetricsVerbosity::NONE, 0);
    }

    void SimulatePulsesNoMetrics() 
    {
        Simulate(Readout::PULSES, MetricsVerbosity::NONE, 0);
    }

    void Summarize(bool flag)
    {
        summarize_ = flag;
    }
};

}}

