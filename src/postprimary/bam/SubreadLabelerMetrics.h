#ifndef SEQUEL_SUBREADLABELERMETRICS_H
#define SEQUEL_SUBREADLABELERMETRICS_H

// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#include <unordered_map>
#include <bazio/RegionLabel.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

struct ControlMetrics
{
    bool called = 0;
    bool isControl = 0;
    uint32_t controlReadLength = 0;
    float controlReadScore = 0;

    size_t EstimatedBytesUsed() const { return sizeof(ControlMetrics); }
};

struct AdapterMetrics
{
    bool called = 0;
    bool corrected = 0;
    bool hasStem = 0;
    int32_t numFound = 0;
    int32_t numFoundSensitive = 0;
    int32_t numFoundPalindrome = 0;
    int32_t numRemoved = 0;
};

struct RegionLabels
{
    std::vector <RegionLabel> adapters;
    std::vector <RegionLabel> barcodes;
    std::vector <RegionLabel> filtered;
    std::vector <RegionLabel> lq;
    std::vector <RegionLabel> inserts;
    RegionLabel hqregion;
    std::vector <uint8_t> cxTags;
    std::unordered_map<size_t, size_t> base2frame;
};

}}}

#endif //SEQUEL_SUBREADLABELERMETRICS_H
