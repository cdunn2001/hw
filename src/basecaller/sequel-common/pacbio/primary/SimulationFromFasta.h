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

#include <cstdlib>
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
#include <pacbio/primary/PrimaryToBaz.h>
#include <pacbio/primary/FastaEntry.h>
#include <pacbio/primary/FastaUtilities.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/SimulationRNG.h>
#include <pacbio/primary/Simulation.h>

namespace PacBio {
namespace Primary {

/// Simulates internal mode BAZ with sequences from a given FASTA file.
/// FASTA sequences are split into chunks.
class SimulationFromFasta
{
public: // structors
    SimulationFromFasta(const std::string& fileName,
                        const std::string& fastaFileName,
                        const Readout readout,
                        const std::string& chipLayoutName,
                        const bool silent,
                        const std::vector<uint32_t>& zmwNumbers,
                        const int numZmws = 0)
        : fileName_(fileName)
        , readout_(readout)
        , chipLayoutName_(chipLayoutName)
        , silent_(silent)
        , numZmws_(numZmws)
    {
        sequences_ = FastaUtilities::ParseSingleFastaFile(fastaFileName);
        numSequences_ = sequences_.size();
        if (numZmws_ == 0)
            numZmws_ = numSequences_;

        // Generate ZMW numbers.
        int z = 0;
        for (int i = 0, h = 1; i < numZmws_; ++h)
            for (int j = 0; j < 1024 && i < numZmws_; ++i, ++j)
                zmwNumbers_.emplace_back(zmwNumbers[z++]);


        Simulate();
    }
    // Default constructor
    SimulationFromFasta() = delete;
    // Move constructor
    SimulationFromFasta(SimulationFromFasta&&) = delete;
    // Copy constructor
    SimulationFromFasta(const SimulationFromFasta&) = delete;
    // Move assignment operator
    SimulationFromFasta& operator=(SimulationFromFasta&& rhs) noexcept = delete;
    // Copy assignment operator
    SimulationFromFasta& operator=(const SimulationFromFasta&) = delete;

private: // data
    const std::string fileName_;
    std::vector<FastaEntry> sequences_;

    float frameRate_ = 100;
    int bps_ = 5;
    int superChunksFrames_ = 16384/2;
    int hFMetricsFrames = 1024;
    int mFMetricFrames = 4096;
    Readout readout_;
    std::string chipLayoutName_;
    bool silent_;
    SimulationRNG rng;
    int numZmws_;
    std::vector<uint32_t> zmwNumbers_;
    int numSequences_;

private:
    void Simulate()
    {
        const bool internal = readout_ == Readout::PULSES;
        const auto metricsVerbosity = internal ? MetricsVerbosity::HIGH : MetricsVerbosity::MINIMAL;

        int seconds_ = std::round(superChunksFrames_ / frameRate_);
        const auto chipLayout = ChipLayout::Factory(chipLayoutName_);
        FileHeaderBuilder fhb("m00001_052415_013000",
                              frameRate_,
                              frameRate_*60*60*3,
                              readout_,
                              metricsVerbosity,
                              generateExperimentMetadata(chipLayoutName_),
                              generateBasecallerConfig(chipLayoutName_.find("Sequ") == 0 ? "Sequel" : "Spider"),
                              zmwNumbers_,
                              std::vector<uint32_t>(),
                              hFMetricsFrames,
                              mFMetricFrames,
                              superChunksFrames_,
                              chipLayout->GetChipClass(),
                              false,  // spiderOnSequel
                              true, // newBazFormat
                              true, // half-float
                              false  // realtimeActivityLabels
        );

        const size_t basesPerSuperChunk = bps_ * seconds_;
        bool superChunksLeft = false;
        size_t superChunkCounter = 0;
        const int numMetrics = 8;

        BazWriter<SequelMetricBlock> writer(fileName_, fhb, PacBio::Primary::BazIOConfig{}, silent_);

        std::vector<uint64_t> currentPulseFrames(numZmws_, 0);
        std::vector<uint64_t> currentBaseFrames(numZmws_, 0);

        do {
            const auto now = std::chrono::high_resolution_clock::now();
            superChunksLeft = false;
            for (int z = 0; z < numZmws_; ++z)
            {
                const auto& s = sequences_.at(z % numSequences_);

                std::string sub = "";
                if (superChunkCounter * basesPerSuperChunk <= s.sequence.size())
                    sub = s.sequence.substr(superChunkCounter * basesPerSuperChunk, basesPerSuperChunk);

                size_t numEvents = sub.size();
                if (numEvents == basesPerSuperChunk) superChunksLeft = true;


                Basecall* basecall = rng.SimulateBaseCalls(sub, &numEvents, &currentPulseFrames[z], &currentBaseFrames[z], readout_);
                if (numEvents < sub.size())
                    throw std::runtime_error("Less pulses than bases: " + std::to_string(numEvents) + " < " + std::to_string(sub.size()));
                std::vector<SequelMetricBlock> hfMetric = rng.SimulateHFMetrics(numMetrics);

                for (int jj = 0; jj < numMetrics; ++jj)
                {
                    auto basesPerMBlock = sub.size() / numMetrics;
                    if (jj == -1)
                        basesPerMBlock += sub.size() % numMetrics;
                    hfMetric[jj].NumBasesA(basesPerMBlock/4);
                    hfMetric[jj].NumBasesC(basesPerMBlock/4);
                    hfMetric[jj].NumBasesG(basesPerMBlock/4);
                    hfMetric[jj].NumBasesT(basesPerMBlock - basesPerMBlock/4 * 3);
                }

                for (int jj = 0; jj < numMetrics-1; ++jj)
                {
                    hfMetric[jj].NumPulses(numEvents / numMetrics);
                }
                hfMetric.back().NumPulses(numEvents / numMetrics + numEvents % numMetrics);

                writer.AddZmwSlice(basecall, numEvents, std::move(hfMetric), z);
                free(basecall);
            }
            // Timing::PrintTime(now, "SimulationFromFasta");
            writer.Flush();
            superChunkCounter++;

            if (!silent_) Timing::PrintTime(now, "Simulation");
            
            while (writer.BufferSize() > 1)
            {
                if (!silent_) std::cerr << "Wait" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }

        } while (superChunksLeft);
        writer.WaitForTermination();
    }
};

}}


