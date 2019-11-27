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

#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>
#include <zlib.h>

#include <pacbio/primary/SequenceUtilities.h>

namespace PacBio {
namespace Primary {

/// Represents a fasta sequence and its ID.
struct FastaEntry
{
public:
    FastaEntry(const std::string& idArg, const std::string& sequenceArg,
               const uint32_t numberArg=0)
        : id(idArg)
        , sequence(sequenceArg)
        , number(numberArg)
    {}

    FastaEntry(const char* const idArg, const char* const sequenceArg,
               const uint32_t numberArg)
        : id(std::string(idArg))
        , sequence(std::string(sequenceArg))
        , number(numberArg)
    {}
    // Move constructor
    FastaEntry(FastaEntry&&) = default;
    // Copy constructor
    FastaEntry(const FastaEntry&) = default;
    // Move assignment operator
    FastaEntry& operator=(FastaEntry&&) = default;
    // Copy assignment operator
    FastaEntry& operator=(const FastaEntry&) = default;
    ~FastaEntry() = default;

public:
    std::string id;
    std::string sequence;
    uint32_t number = 0;

public:
    void ReverseCompl()
    { sequence = SequenceUtilities::ReverseCompl(sequence); }

    int Length() const
    { return sequence.size(); }
};

}}
