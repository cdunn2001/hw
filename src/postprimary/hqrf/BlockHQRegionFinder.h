// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#include <vector>

#include <bazio/BlockActivityLabels.h>
#include <pacbio/primary/HQRFMethod.h>

#include "HQRegionFinder.h"
#include "HQRegionFinderModels.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// The state of the HQRF HMM
enum HQRFState
{
    PRE = 0,     // Possible multiple polymerase activity before HQR
    HQ,          // In the HQR
    POST_QUIET,  // After the HQR, no polymerase activity
    POST_ACTIVE, // After the HQR, multiple polymerase activity resumes

    NUM_STATES
};

/// Evaluate a polynomial of the form y = ax^2 + bx + c
/// This is in this namespace for testability (vs anon)
float evaluatePolynomial(const std::vector<float>& coeff, float x);

// Class for finding the HQ region depending on the metric data.  Templated on
// the HQRMMethod, which mainly is used to determine how the ActivityLabels
// are generated
template <HQRFMethod Method>
class BlockHQRegionFinder final : public HQRegionFinder
{
    class UnknownFeatureException : public std::runtime_error
    {
    public:
        UnknownFeatureException() : std::runtime_error(
            "HQRF is missing feature requried by model") {}
    };

public: // structors
    // Primary constructor
    BlockHQRegionFinder(float frameRate, float snrThresh, bool ignoreBazAL);

    // No copy/move
    BlockHQRegionFinder(BlockHQRegionFinder&&) = delete;
    BlockHQRegionFinder(const BlockHQRegionFinder&) = delete;
    BlockHQRegionFinder& operator=(BlockHQRegionFinder&&) = delete;
    BlockHQRegionFinder& operator=(const BlockHQRegionFinder&) = delete;
    
    ~BlockHQRegionFinder() override = default;

public: /// Low-level calls, exposed for testing purposes

    /// Label the MF intervals based on the MF metrics
    std::vector<ActivityLabeler::Activity> LabelActivities(const BlockLevelMetrics& metrics) const;

    /// Find the HQ region, given window activity labels.  Interval
    /// returned is 0-based interval [start, end) indexing into the MF
    /// intervals vector input.  (Method exposed for testing purposes)
    std::pair<int, int> FindBlockHQRegion(const std::vector<ActivityLabeler::Activity>& activitySeq,
                                          std::vector<HQRFState>* stateSequence = nullptr,
                                          float* loglikelihood = nullptr) const;

private:
    std::pair<int, int> LabelRegions(
            const std::vector<ActivityLabeler::Activity>& activitySeq,
            const float transitionProb[NUM_STATES][NUM_STATES],
            const float emissionProb[NUM_STATES][ActivityLabeler::LABEL_CARDINALITY],
            std::vector<HQRFState>* stateSequence,
            float* loglikelihood) const;

public: /// Client API
    /// Finds the HQR as a range in pulses coordinates.
    std::pair<size_t, size_t> FindHQRegion(const BlockLevelMetrics& metrics, const EventData& zmw) const override;

private: // Members
    float frameRate_;
    float snrThresh_;
    bool ignoreBazAL_;
};

}}} // ::PacBio::Primary::Postprimary


