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

#define BYTETOBINARY(byte)  \
  (byte & 0x80 ? 1 : 0), \
  (byte & 0x40 ? 1 : 0), \
  (byte & 0x20 ? 1 : 0), \
  (byte & 0x10 ? 1 : 0), \
  (byte & 0x08 ? 1 : 0), \
  (byte & 0x04 ? 1 : 0), \
  (byte & 0x02 ? 1 : 0), \
  (byte & 0x01 ? 1 : 0) 

#include <pacbio/primary/PacketFieldName.h>

#include <iostream>

namespace PacBio {
namespace Primary {

/// \brief   Describes a single packet field.
/// \details Stores all necessary information to parse a byte and bitshift 
///          the correct bits, where the sought after information is saved.
///          Also encodes if this field has a potential extension and its size.
struct PacketField
{    
public:
    PacketField() = default;

    PacketField(const PacketFieldName fieldNameArg) 
        : fieldName(fieldNameArg) {}

    PacketField(const PacketFieldName fieldNameArg, 
                const uint8_t fieldBitSizeArg) 
        : fieldName(fieldNameArg)
        , fieldBitSize(fieldBitSizeArg) {}

    PacketField(const PacketFieldName fieldNameArg, 
                const uint8_t fieldBitSizeArg,
                const uint8_t fieldEscapeValueArg,
                const PacketFieldName extensionNameArg,
                const uint8_t extensionBitSizeArg) 
        : fieldName(fieldNameArg)
        , extensionName(extensionNameArg)     
        , hasFieldEscape(true) 
        , fieldBitSize(fieldBitSizeArg)
        , fieldEscapeValue(fieldEscapeValueArg)   
        , extensionBitSize(extensionBitSizeArg) {}

    // Move constructor
    PacketField(PacketField&&) = default;
    // Copy constructor
    PacketField(const PacketField&) = default;
    // Move assignment operator
    PacketField& operator=(PacketField&&) = default;
    // Copy assignment operator
    PacketField& operator=(const PacketField&) = default;
    // Destructor
    ~PacketField() = default;

public:
    PacketFieldName fieldName        = PacketFieldName::GAP;
    PacketFieldName extensionName    = PacketFieldName::GAP;
    bool            hasFieldEscape   = false;
    uint8_t         fieldBitSize     = 0;
    uint8_t         fieldBitMask     = 0;
    uint8_t         fieldBitShift    = 0;
    uint8_t         fieldEscapeValue = 0;
    uint8_t         extensionBitSize = 0;

public:
    void Print() const
    { 
        std::cout << fieldName.toString() << ":" << std::to_string(fieldBitSize);
        if (hasFieldEscape)
            std::cout << "::" << extensionName.toString() 
                << ":" << std::to_string(extensionBitSize)
                << "::" << std::to_string(MaxBitSize());
        else
            printf ("::%d%d%d%d%d%d%d%d:%d", BYTETOBINARY(fieldBitMask), fieldBitShift);
        std::cout << std::endl; 
    }

    uint8_t MaxBitSize() const
    { return fieldBitSize > extensionBitSize ? fieldBitSize : extensionBitSize; }
};

}}
