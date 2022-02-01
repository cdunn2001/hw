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

#include <string>

#include <bazio/RegionLabelType.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Describes the surrounding environment of a subread.
struct SubreadContext
{
public:
    static const int UNSET = -1;
public:
    SubreadContext() = default;

    SubreadContext(int beginArg, int endArg) 
        : begin(beginArg)
        , end(endArg) {}

    SubreadContext(int beginArg, int endArg, RegionLabelType labelArg)
        : begin(beginArg)
        , end(endArg)
        , label(labelArg) {}

public:
    int  begin            = -1;
    int  end              = -1;
    int  barcodeIdBefore  = -1;
    int  barcodeIdAfter   = -1;
    int  adapterIdBefore  = -1;
    int  adapterIdAfter   = -1;
    bool adapterBefore    = false;
    bool adapterAfter     = false;
    RegionLabelType label = RegionLabelType::NOT_SET;
    // Adapter calling statistics
    float adapterAccuracyBefore      = 0.0f;
    float adapterAccuracyAfter       = 0.0f;
    int   adapterFlankingScoreBefore = -1;
    int   adapterFlankingScoreAfter  = -1;
    bool hasStemBefore = false;
    bool hasStemAfter = false;
};

}}} // ::PacBio::Primary::Postprimary

