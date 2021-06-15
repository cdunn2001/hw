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

#include <pacbio/smrtdata/Readout.h>

#include "Codec.h"
#include "MemoryBuffer.h"

namespace PacBio {
namespace SmrtData {
class Pulse;
class Basecall;
enum class NucleotideLabel : uint8_t;
}} // ::PacBio::Primary

namespace PacBio {
namespace Primary {


template <typename TMetric>
class PrimaryToBaz
{
public: // static
    // constexpr std::vector<uint8_t> bla() const;
public: // structors

    // Default constructor
    PrimaryToBaz(const uint32_t maxNumZmws, SmrtData::Readout readout);
    // Move constructor
    PrimaryToBaz(PrimaryToBaz&&) = delete;
    // Copy constructor
    PrimaryToBaz(const PrimaryToBaz&) = delete;
    // Move assignment operator
    PrimaryToBaz& operator=(PrimaryToBaz&& rhs) noexcept = delete;
    // Copy assignment operator
    PrimaryToBaz& operator=(const PrimaryToBaz&) = delete;

    ~PrimaryToBaz() = default;

public: // methods
     
    Primary::MemoryBufferView<uint8_t> SliceToPacket(const SmrtData::Basecall* basecall,
                                              const uint16_t numEvents,
                                              const uint32_t zmwNum,
                                              Primary::MemoryBuffer<uint8_t>& allocator,
                                              uint32_t* packetByteStreamSize,
                                              uint16_t* numIncorporatedEvents);

private: // data
    std::vector<uint64_t> currentPulseFrames_;
    std::vector<uint64_t> currentBaseFrames_;
    SmrtData::Readout readout_;

    Codec codec{};
private:
    Primary::MemoryBufferView<uint8_t> SliceToPulsePacket(
            const SmrtData::Basecall* basecall,
            const uint16_t numEvents, 
            const uint32_t zmwNum,
            Primary::MemoryBuffer<uint8_t>& allocator,
            uint32_t* packetByteStreamSize);

    Primary::MemoryBufferView<uint8_t> SliceToBasePacket(
            const SmrtData::Basecall* basecall,
            const uint16_t numEvents, 
            const uint32_t zmwNum,
            Primary::MemoryBuffer<uint8_t>& allocator,
            uint32_t* packetByteStreamSize,
            uint16_t* numIncorporatedEvents);

    Primary::MemoryBufferView<uint8_t> SliceToBasePacketMinimal(
            const SmrtData::Basecall* basecall,
            const uint16_t numEvents,
            const uint32_t zmwId,
            Primary::MemoryBuffer<uint8_t>& allocator,
            uint32_t* packetByteStreamSize,
            uint16_t* numIncorporatedEvents);

    static inline uint8_t NucleotideLabelToBaz(const SmrtData::NucleotideLabel& label)
    { return static_cast<uint8_t>(label); }
    static_assert(static_cast<int>(SmrtData::NucleotideLabel::A) == 0,"A");
    static_assert(static_cast<int>(SmrtData::NucleotideLabel::C) == 1,"C");
    static_assert(static_cast<int>(SmrtData::NucleotideLabel::G) == 2,"G");
    static_assert(static_cast<int>(SmrtData::NucleotideLabel::T) == 3,"T");
    static_assert(static_cast<int>(SmrtData::NucleotideLabel::N) == 4,"N");
};

}}
