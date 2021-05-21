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

#include <cstdlib>
#include <stdexcept>

#include "RegionLabelType.h"

namespace PacBio {
namespace Primary {

/// Single label that annotates a region of a read.
struct RegionLabel
{
public:
    RegionLabel() = default;

    RegionLabel(const int beginArg, const int endArg, const int scoreArg,
                RegionLabelType typeArg)
        : begin(beginArg)
        , end(endArg)
        , score(scoreArg)
        , type(typeArg){};

    RegionLabel(const int beginArg, const int endArg, const int scoreArg, 
                const int sequenceIdArg, const RegionLabelType typeArg)
        : begin(beginArg)
        , end(endArg)
        , score(scoreArg)
        , sequenceId(sequenceIdArg)
        , type(typeArg){};

    // Move constructor
    RegionLabel(RegionLabel&&) = default;
    // Copy constructor is deleted!
    RegionLabel(const RegionLabel&) = default;
    // Move assignment operator
    RegionLabel& operator=(RegionLabel&&) = default;
    // Copy assignment operator is deleted!
    RegionLabel& operator=(const RegionLabel&) = delete;
    // Destructor
    ~RegionLabel() = default;
    
public: // static methods
    static void CutRegionLabels(RegionLabel* first, RegionLabel* second)
    {
        if (first->begin > second->begin || first->end > second->end)
        {
            auto min = std::min(std::min(first->begin, first->end), std::min(second->begin, second->end));
            auto max = std::max(std::max(first->begin, first->end), std::max(second->begin, second->end));
            first->begin = min;
            second->end = max;
            auto overlap = (max-min)/2;
            first->end = min + overlap;
            second->begin = first->end;
        }
        if (first->end > second->begin)
        {
            // auto fb = first->begin;
            // auto fe = first->end;
            // auto sb = second->begin;
            // auto se = second->end;
            int overlap = std::abs(first->end - second->begin)/2;
            first->end -= overlap;
            second->begin = first->end;
            if (first->begin > first->end || second->begin > second->end)
            {
                throw std::runtime_error("Region cutting problem ");// + std::to_string(fb) + " " + std::to_string(fe) + " " + std::to_string(sb) + " " + std::to_string(se) + " " + std::to_string(overlap));
            }
        }
        if (first->end == second->begin)
        {
            first->callDownstreamFlank = false;
            second->callUpstreamFlank = false;
        }
    }

public:
    inline int Length() const
    { return end - begin; }

public:
    int pulseBegin = 0;
    int pulseEnd = 0;
    int begin = 0;
    int end = 0;
    double score = -1;
    int sequenceId = -1;
    RegionLabelType type = RegionLabelType::NOT_SET;
    bool save = true;
    bool callUpstreamFlank = true;
    bool callDownstreamFlank = true;
    int flankingScore = -1;
    float accuracy = 0.0f;
    bool hasStem = false;
};

inline bool RegionBeginComparer(const RegionLabel& lhs, const RegionLabel& rhs) 
{ 
    return lhs.begin < rhs.begin; 
}

}}
