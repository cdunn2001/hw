// Copyright (c) 2015-2018, Pacific Biosciences of California, Inc.
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

// Programmer: Martin Smith

#include <bazio/PacketFieldName.h>

#include "InsertFinder.h"
#include "InsertState.h"

#include <algorithm>
#include <cmath>

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Insert Finder Emissions are pw 0 to 9 (pw = min(pw, 9))
static constexpr size_t MAX_BURST_PW = 10;

namespace
{

template<typename T>
size_t argmax(T begin, T end)
{
    return std::max_element(begin, end) - begin;
}

}

std::vector<uint32_t> ObservePulseWidths(const BazIO::BazEventData& packets)
{
    // Pulse widths from the BAZ file are for all pulses so need
    // to check that we are returning only pulse widths from
    // pulses called as bases.
    std::vector<uint32_t> pulseWidths;
    const auto& pws = packets.PulseWidths();
    const auto& isBase = packets.IsBase();
    assert(isBase.size() == pws.size());

    pulseWidths.reserve(pws.size());

    for (size_t pw = 0; pw < pws.size(); pw++)
    {
        if (isBase[pw])
            pulseWidths.push_back(std::min(static_cast<uint32_t>(MAX_BURST_PW), pws[pw]));
    }
    return pulseWidths;
}

std::vector<InsertState> BurstInsertFinder::ClassifyInserts(const BazIO::BazEventData& packets) const
{
    const auto& pulseWidths = ObservePulseWidths(packets);

    // Observed pulse widths are from pulses called as bases.
    size_t numBases = pulseWidths.size();

    if (numBases == 0)
        return std::vector<InsertState>(packets.NumEvents(), InsertState::EX_SHORT_PULSE);

    // HMM courtesy of A. Wenger. The current burst identifier HMM
    // only identifies BASE and BURST_PULSE so the transition
    // probabilities of the other states are set to 0. The
    // reference implementation can be founder under
    // Sequel/ppa/doc/hmmCleanBursts.

    constexpr auto NUM_STATES = static_cast<std::size_t>(InsertState::NUM_STATES);
    const float transitionProb[NUM_STATES][NUM_STATES] =
    {
        { 0.999795729f,          0,       0.000204271f,          0 }, // BASE ->
        {            0,          0,                  0,          0 }, // EX_SHORT_PULSE ->
        { 0.020527268f,          0,       0.979472732f,          0 }, // BURST_PULSE ->
        {            0,          0,                  0,          0 }  // PAUSE_PULSE ->
    };

    const float emissionProb[NUM_STATES][MAX_BURST_PW] =
    {
        { 0.00089901f, 0.02193594f, 0.05354958f, 0.06332271f, 0.06570604f,
          0.06429624f,  0.0610961f, 0.05710235f,  0.0526792f, 0.55941284f   }, // PW | BASE
        {           0,           0,           0,           0,          0,
                    0,           0,           0,           0,          0    }, // PW | EX_SHORT_PULSE
        { 0.03688890f, 0.20162510f, 0.32948849f, 0.15870858f, 0.08543165f,
          0.05133312f,  0.0325789f, 0.02274935f,  0.0159825f, 0.06521340f   }, // PW | BURST_PULSE
        {           0,           0,           0,           0,          0,
                    0,           0,           0,           0,          0    }  // PW | PAUSE_PULSE
    };

    // Log-space.
    float logTransitionProb[NUM_STATES][NUM_STATES];
    for (size_t i = 0; i < NUM_STATES; i++)
        for (size_t j = 0; j < NUM_STATES; j++)
            logTransitionProb[i][j] = logf(transitionProb[i][j]);

    float logEmissionProb[NUM_STATES][MAX_BURST_PW];
    for (size_t i = 0; i < NUM_STATES; i++)
        for (size_t j = 0; j < MAX_BURST_PW; j++)
            logEmissionProb[i][j] = logf(emissionProb[i][j]);

    std::vector<std::array<float, NUM_STATES>> dpM(numBases + 1);
    std::vector<std::array<int, NUM_STATES>> tbM(numBases + 1);

    // Initialize starting probabilities.
    dpM[0][static_cast<size_t>(InsertState::BASE)]           = logf(0.99);
    dpM[0][static_cast<size_t>(InsertState::EX_SHORT_PULSE)] = logf(0);
    dpM[0][static_cast<size_t>(InsertState::BURST_PULSE)]    = logf(0.01);
    dpM[0][static_cast<size_t>(InsertState::PAUSE_PULSE)]    = logf(0);

    // Run Viterbi.
    for (size_t p = 1; p <= numBases; p++)
    {
        auto pw = pulseWidths[p-1];
        for (size_t q = 0; q < NUM_STATES; q++)
        {
            std::array<float, NUM_STATES> P;
            for (size_t qp = 0; qp < NUM_STATES; qp++)
                P[qp] = dpM[p-1][qp] + logTransitionProb[qp][q] + logEmissionProb[q][pw-1];

            size_t qMax = argmax(P.begin(), P.end());

            tbM[p][q] = static_cast<uint>(qMax);
            dpM[p][q] = P[qMax];
        }
    }

    auto& lastRow = dpM[numBases];
    size_t q = argmax(std::begin(lastRow), std::end(lastRow));

    std::vector<InsertState> states(packets.NumEvents(), InsertState::EX_SHORT_PULSE);
    const auto& isBase = packets.IsBase();

    // N.B.
    // tbM[0] is uninitialized, and tbM[1] is basically meaningless (it's the
    // transition back to the initial conditions we used to launch the viterbi
    // algorithm).  Re-write all of tbM[1] with InsertState::NUM_STATE.  This
    // ensures that the last iteration of the loop below has a state transition
    // and outputs a series of labels.  This causes less cleanup and redundant
    // logic to handle the last series after the loop.
    std::fill(tbM[1].begin(), tbM[1].end(), static_cast<int>(InsertState::NUM_STATES));

    InsertState currState = static_cast<InsertState>(q);
    int64_t rightIdx = numBases-1;
    int64_t pulseIdx = packets.NumEvents(); // start outside array, will decrement before use
    for (int64_t baseIndex = numBases-1; baseIndex >= 0; baseIndex--)
    {
        q = tbM[baseIndex+1][q];

        // Only write to the `state` array at a state transition, so we know
        // the full length of the sequence.
        if (static_cast<InsertState>(q) != currState)
        {
            auto sameStateCount = rightIdx - baseIndex + 1;
            // If we've got a burst phase that is too short, keep them as bases
            if (currState == InsertState::BURST_PULSE)
            {
                if (sameStateCount < minInsertionLength_)
                    currState = InsertState::BASE;
            }

            // Write out the state labels, being careful to leave things
            // that were pulses in the baz file as `EX_SHORT_PULSE`
            while (sameStateCount > 0)
            {
                pulseIdx--;
                assert(pulseIdx >= 0);
                if (isBase[pulseIdx])
                {
                    states[pulseIdx] = currState;
                    sameStateCount--;
                }
            }

            rightIdx = baseIndex - 1;
            currState = static_cast<InsertState>(q);
        }
    }

    return states;
}

std::vector<InsertState> SimpleInsertFinder::ClassifyInserts(const BazIO::BazEventData& packets) const
{
    if (packets.Internal())
    {
        std::vector<InsertState> ret;
        ret.reserve(packets.NumEvents());

        const auto& isBase = packets.IsBase();
        std::transform(isBase.begin(), isBase.end(), std::back_inserter(ret),
                [](bool base) { return base ? InsertState::BASE : InsertState::EX_SHORT_PULSE; });

        return ret;
    }
    else
    {
        return std::vector<InsertState>(packets.NumEvents(), InsertState::BASE);
    }
}

std::unique_ptr<InsertFinder>
InsertFinderFactory(const UserParameters& user)
{
    if (user.noClassifyInsertions)
    {
        return std::make_unique<SimpleInsertFinder>();
    }
    else
    {
        return std::make_unique<BurstInsertFinder>(user.minInsertionLength);
    }
}

}}} // ::PacBio::Primary::Postprimary

