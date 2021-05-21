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

#include <vector>
#include <memory>

#include <bazio/FastaEntry.h>

#include <postprimary/adapterfinder/AdapterLabeler.h>
#include <postprimary/application/PpaAlgoConfig.h>
#include <postprimary/application/UserParameters.h>
#include <postprimary/controlfinder/ControlFilter.h>
#include <postprimary/stats/ProductivityMetrics.h>

#include "EventData.h"
#include "ResultPacket.h"
#include "RuntimeMetaData.h"
#include "SubreadLabelerMetrics.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Label a given read using AF, CF, and HQRF.
class SubreadLabeler
{
public: // structors
    // Default constructor
    SubreadLabeler(const std::shared_ptr<UserParameters>& user,
                   const std::shared_ptr<RuntimeMetaData>& rmd,
                   const std::shared_ptr<PpaAlgoConfig>& ppaAlgoConfig);
    // Move constructor
    SubreadLabeler(SubreadLabeler&&) = delete;
    // Copy constructor
    SubreadLabeler(const SubreadLabeler&) = delete;
    // Move assignment operator
    SubreadLabeler& operator=(SubreadLabeler&&) = delete;
    // Copy assignment operator
    SubreadLabeler& operator=(const SubreadLabeler&) = delete;
    // Destructor
    ~SubreadLabeler() = default;

public: // non-modifying methods

    ControlMetrics CallControls(
        const ProductivityInfo& pinfo,
        const EventData& events, const RegionLabel& hqregion,
        bool isControl) const;

    // \brief Calls adapters present in hqregion basecalls, unless the read is
    // a control
    // \returns a tuple: vector of called adapters, Metrics about how those
    // adapters were called, and an expanded hqregion that includes any adapter
    // calls on the hqregion boundary.
    std::tuple<std::vector<RegionLabel>, AdapterMetrics, RegionLabel>
    CallAdapters(const EventData& events,
                 const RegionLabel& hqregion,
                 bool isControl,
                 bool emptyPulseCoordinates=false) const;

    /// \brief Splits a single polymeraseread to its subreads.
    /// \return Returns a vector of ResultPacket that contain subreads.
    std::vector<ResultPacket>
    ReadToResultPacket(
        const ProductivityInfo& pinfo, const EventData& events,
        const RegionLabels& regions, const double barcodeScore,
        const ControlMetrics& controlMetrics) const;

    bool PerformAdapterFinding() const
    { return performAdapterFinding_; };

    bool PerformControlFiltering() const
    { return performControlFiltering_; };

private:
    std::shared_ptr<UserParameters>  user_;
    std::shared_ptr<RuntimeMetaData> rmd_;
    std::shared_ptr<PpaAlgoConfig>   ppaAlgoConfig_;
    bool performAdapterFinding_  = false;
    bool performControlFiltering_ = false;

    std::shared_ptr<std::vector<PacBio::Primary::FastaEntry>> adapterList_;
    std::shared_ptr<std::vector<PacBio::Primary::FastaEntry>> rcAdapterList_;
    std::shared_ptr<std::vector<PacBio::Primary::FastaEntry>> controlList_;
    std::shared_ptr<std::vector<PacBio::Primary::FastaEntry>> controlAdapterList_;

    std::unique_ptr<ScoringScheme>  palindromeScorer_;
    std::unique_ptr<AdapterLabeler> sensitiveAdapterLabeler_;
    std::unique_ptr<AdapterLabeler> rcAdapterLabeler_;
    std::unique_ptr<AdapterLabeler> adapterLabeler_;
    std::unique_ptr<ControlFilter>  controlFilter_;

private:
    // non-modifying methods
    ControlMetrics FilterControls(const RegionLabel& hqregion, const EventData& events) const;

private:
    // modifying methods
    void InitFromPpaAlgoConfig();
    void InitAdapterFinding();
    void InitControlFiltering();
};

}}} // ::PacBio::Primary::Postprimary

