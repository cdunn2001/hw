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

#include <stdexcept>

#include <postprimary/bam/Platform.h>

#include "ScoringScope.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// SW and NW scoring scheme
struct ScoringScheme
{
public:
    ScoringScheme() = default;

    ScoringScheme(const Platform platform, const ScoringScope scope)
    {
        Init(platform, scope);
    };
    // Move constructor
    ScoringScheme(ScoringScheme&&) = default;
    // Copy constructor
    ScoringScheme(const ScoringScheme&) = default;
    // Move assignment operator
    ScoringScheme& operator=(ScoringScheme&&) = default;
    // Copy assignment operator
    ScoringScheme& operator=(const ScoringScheme&) = default;

    ~ScoringScheme() = default;

public:
    int32_t matchScore = INT32_MIN;
    int32_t mismatchPenalty = INT32_MIN;
    int32_t insertionPenalty = INT32_MIN;
    int32_t deletionPenalty = INT32_MIN;
    int32_t branchPenalty = INT32_MIN;

public:
    void Init(const Platform platform, const ScoringScope scope)
    {
        switch(platform)
        {
            case Platform::RSII:
            case Platform::SEQUEL:
            case Platform::SEQUELII:
            case Platform::MINIMAL:
            {
                switch (scope)
                {
                    case ScoringScope::ADAPTER_FINDING:
                        matchScore = 4;
                        mismatchPenalty = -13;
                        insertionPenalty = -7;
                        deletionPenalty = -7;
                        branchPenalty = -4;
                        break;
                    case ScoringScope::FLANKING_REGIONS:
                        matchScore = 5;
                        mismatchPenalty = -9;
                        insertionPenalty = -5;
                        deletionPenalty = -5;
                        branchPenalty = 0; // Not used
                        break;
                    case ScoringScope::BARCODE_CALLING:
                        matchScore = 4;
                        mismatchPenalty = -13;
                        insertionPenalty = -7;
                        deletionPenalty = -7;
                        branchPenalty = -4;
                        break;
                }
            }
                break;
            case Platform::NONE:
            default:
                throw std::runtime_error("Platform not specified!");
                break;
        }
    }
};

}}} // ::PacBio::Primary::Postprimary
