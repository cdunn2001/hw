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

#include <pacbio/primary/MetricFieldName.h>

namespace PacBio {
namespace Primary {

/// A single metric field with its name and bit size.
struct MetricField
{
public:
     inline static std::string NameToString(MetricFieldName f) { return f.toString(); }

public:
    MetricField() = default;
    MetricField(const MetricFieldName& fieldNameArg, 
                uint8_t fieldBitSizeArg,
                bool fieldSignedArg,
                uint16_t fieldScalingFactorArg)
        : fieldName(fieldNameArg)
        , fieldBitSize(fieldBitSizeArg)
        , fieldSigned(fieldSignedArg)
        , fieldScalingFactor(fieldScalingFactorArg) {}
    // Move constructor
    MetricField(MetricField&&) = default;
    // Copy constructor
    MetricField(const MetricField&) = default;
    // Move assignment operator
    MetricField& operator=(MetricField&&) = default;
    // Copy assignment operator
    MetricField& operator=(const MetricField&) = default;
    // Destructor
    ~MetricField() = default;

public:
    MetricFieldName fieldName          = MetricFieldName::GAP;
    uint8_t         fieldBitSize       = 0;
    bool            fieldSigned        = false;
    uint16_t        fieldScalingFactor = 1;

public:
    void Print()
    { 
        std::cout << NameToString(fieldName) << ":" 
                  << std::to_string(fieldBitSize) << std::endl; 
    }
};

}}
