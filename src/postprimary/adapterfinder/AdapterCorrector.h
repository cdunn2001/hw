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

#include <bazio/RegionLabel.h>

#include <postprimary/alignment/ScoringScheme.h>
#include "AdapterLabeler.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

// Detection stages:

/// Look for subreads that are roughly 2X longer than they should be.
/// Adapters should always be non-overlapping and sorted by location
/// Returns a list of subreads indices
std::vector<int> FlagSubreadLengthOutliers(
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqRegion,
    float lengthThreshold=1.8);

/// Look for palindromes around adapters, and make sure there are two
/// distinct pairs of sequences (one for each end of the insert).
/// Adapters should always be non-overlapping and sorted by location
/// Returns a list of subreads indices
std::vector<int> FlagAbsentAdapters(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const ScoringScheme& scoreScheme,
    int flankSize=50,
    int flankScore=36);

// Correction stages:

/// Flagged subreads likely contain an adapter that we missed. They may also
/// contain a missing adapter, but that will be detected and fixed
/// separately.
/// Flagged is a sorted list of subread indices
std::vector<RegionLabel> SensitiveAdapterSearch(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqRegion,
    const std::vector<int>& flagged,
    const AdapterLabeler& sensitiveAdapterLabeler,
    const AdapterLabeler& rcAdapterLabeler);


/// Flagged subreads likely contain two passes and a missing adapter.
/// Flagged is a sorted list of subread indices
std::vector<RegionLabel> PalindromeSearch(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqRegion,
    bool removeOnly = false);

/// Remove adapters that split a subread that would otherwise be ~median
/// length
std::vector<RegionLabel> RemoveFalsePositives(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqregion);

// Utility functions:

/// Find the center of a given palindrome using a trie
std::vector<int> PalindromeTrieCenter(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    int hqrOffset,
    int kmerSize,
    int centerSampleSize);

std::pair<int, int> HashKmer(
    const std::string& bases,
    int pos,
    int kmerSize,
    const std::vector<int>& powlut);

std::pair<int, int> UpdateKmerHash(
    const std::string& bases,
    int pos,
    int kmerSize,
    const std::vector<int>& powlut,
    int fdIndex,
    int rcIndex);

struct SensitiveAdapterLabeler {
    static constexpr float minSoftAccuracy = 0.65;
    static constexpr float minHardAccuracy = 0.5;
    static constexpr int minFlankingScore = 214;
    static constexpr bool localAlnFlanking = true;
    static constexpr int flankLength = 250;
};

}}} // ::PacBio::Primary::Postprimary
