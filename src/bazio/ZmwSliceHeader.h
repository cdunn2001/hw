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

namespace PacBio {
namespace Primary {

/// Describes a ZMW_SLICE_HEADER and its properties.
struct ZmwSliceHeader
{
    uint64_t offsetPacket;         //  8 B
    uint32_t zmwIndex;             // 12 B
    uint32_t packetsByteSize;      // 16 B
    uint16_t numEvents;            // 18 B
    uint16_t numHFMBs;             // 20 B number of high-frequency metric blocks
    uint8_t  numMFMBs;             /* 21 B number of medium-frequency metric blocks
                                           CANNOT be larger than 15 */
    uint8_t  numLFMBs;             /* 22 B number of low-frequency metric blocks
                                           CANNOT be larger than 15 */

    static constexpr size_t SizeOf() 
    { 
        return sizeof(offsetPacket)  + sizeof(zmwIndex) +
               sizeof(packetsByteSize) +
               sizeof(numEvents) + sizeof(numHFMBs) + sizeof(numMFMBs) + sizeof(numLFMBs); 
    }


};
static_assert(ZmwSliceHeader::SizeOf() == 22, "ZmwSliceHeader is not 22!");

}}
