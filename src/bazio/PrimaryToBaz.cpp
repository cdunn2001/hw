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

#include <algorithm>
#include <errno.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

#include <pacbio/logging/Logger.h>
#include <pacbio/PBException.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/NucleotideLabel.h>
#include <pacbio/smrtdata/Pulse.h>
#include <pacbio/smrtdata/Readout.h>

#include "SmartMemory.h"
#include "PrimaryToBaz.h"
#include "BazWriter.h"

using namespace PacBio::SmrtData;

namespace PacBio {
namespace Primary {

template <typename TMetric>
PrimaryToBaz<TMetric>::PrimaryToBaz(const uint32_t maxNumZmws, Readout readout)
  : readout_(readout)
{
    currentPulseFrames_.resize(maxNumZmws, 0);
    currentBaseFrames_.resize(maxNumZmws, 0);
}

template <typename TMetric>
MemoryBufferView<uint8_t> PrimaryToBaz<TMetric>::SliceToPacket(
        const Basecall* basecall,
        const uint16_t numEvents,
        const uint32_t zmwId,
        MemoryBuffer<uint8_t>& allocator,
        uint32_t* packetByteStreamSize,
        uint16_t* numIncorporatedEvents)
{
    switch (readout_)
    {
        case Readout::PULSES:
            *numIncorporatedEvents = numEvents;
            return SliceToPulsePacket(basecall, 
                                      numEvents,
                                      zmwId,
                                      allocator,
                                      packetByteStreamSize);
        case Readout::BASES:
        case Readout::BASES_WITHOUT_QVS:
            return SliceToBasePacketMinimal(basecall, 
                                            numEvents, 
                                            zmwId,
                                            allocator,
                                            packetByteStreamSize,
                                            numIncorporatedEvents);
        default:
            throw PBException("Not aware of that Readout.");
    }
}

template <typename TMetric>
MemoryBufferView<uint8_t> PrimaryToBaz<TMetric>::SliceToPulsePacket(
        const Basecall* basecall,
        const uint16_t numEvents,
        const uint32_t zmwId,
        MemoryBuffer<uint8_t>& allocator,
        uint32_t* packetByteStreamSize)
{
    std::vector<uint32_t> ipds;
    ipds.reserve(numEvents);

    uint16_t numExtensions16bit = 0;
    uint16_t numExtensions32bit = 0;

    int currentPulseFrame = currentPulseFrames_[zmwId];
    int currentBaseFrame  = currentBaseFrames_[zmwId];
    for (size_t i = 0; i < numEvents; ++i)
    {
        const auto& b = basecall[i];
        const auto& p = b.GetPulse();
        uint32_t ipd;
        if (!b.IsNoCall())
        {
            ipd = p.Start() - currentBaseFrame;
            ipds.emplace_back(ipd);
            currentBaseFrame = p.Start() + p.Width();
            currentPulseFrame = currentBaseFrame;
        }
        else
        {
            ipd = p.Start() - currentPulseFrame;
            ipds.emplace_back(ipd);
            currentPulseFrame = p.Start() + p.Width();
        }
        
        if (ipd >= 255)
            ++numExtensions32bit;
        if (p.Width() >= 255)
            ++numExtensions32bit;
        if (std::round(p.MeanSignal() * 10) >= 255)
            ++numExtensions16bit;
        if (std::round(p.MidSignal() * 10) >= 255)
            ++numExtensions16bit;
    }

    currentPulseFrames_[zmwId] = currentPulseFrame;
    currentBaseFrames_[zmwId]  = currentBaseFrame;

    constexpr
    uint32_t packetByteSize = ( 2 // base
                              + 3 // deletionTag
                              + 3 // substitutionTag
                              + 4 // deletionQV
                              + 4 // substitutionQV
                              + 4 // insertionQV
                              + 4 // mergeQV
                              + 1 // isBase
                              + 1 // isPulse
                              + 3 // label
                              + 3 // alternative label
                              + 4 // labelQV
                              + 4 // alternative Label QV
                              + 8 // IPD
                              + 8 // PW
                              + 8 // PKMID
                              + 8 // PKMEAN
                              // + 8 // PKMID2
                              // + 8 // PKMEAN2
                              )/8;// Bits to bytes
    static_assert(packetByteSize == BazWriter<TMetric>::packetSizeInt,"mismatch in calculated and expected size");
    *packetByteStreamSize = packetByteSize * numEvents 
                            + 2 * numExtensions16bit
                            + 4 * numExtensions32bit;

    MemoryBufferView<uint8_t> buffer = allocator.Allocate(*packetByteStreamSize);
    size_t counter = 0;
    size_t numIPDs = 0;

    for (size_t i = 0; i < numEvents; ++i)
    {
        const auto& b = basecall[i];
        const auto& p = b.GetPulse();
        NucleotideLabel baseLabel = b.Base2();
        bool isBase = !b.IsNoCall();
        bool isPulse;
        uint8_t base;
        
        // In case that there is no base call and the label and altLabel tags
        // are NONE or 'N', we will tag this pulse as non-existent and set 
        // base to the arbitrary number 0 for 'A'. The conversion in StitchedZmw
        // will take care of converting READOUTs to 'n' if is not a base and 
        // not a pulse.
        if (!isBase && (baseLabel == NucleotideLabel::NONE ||
                        baseLabel == NucleotideLabel::N))
        {
            isPulse = false;
            base = 0;
        }
        else
        {
            isPulse = true;
            base = NucleotideLabelToBaz(b.Base2()); 
        }

        //
        // 1. Byte
        if (base > 3) throw PBException("Base OOB");

        const auto delTag = NucleotideLabelToBaz(b.DeletionTag());
        if (b.DeletionTag() == NucleotideLabel::NONE || delTag > 4) 
            throw PBException("DeletionTag OOB");

        const auto subTag = NucleotideLabelToBaz(b.SubstitutionTag());
        if (b.SubstitutionTag() == NucleotideLabel::NONE || subTag > 4) 
            throw PBException("SubstitutionTag OOB");

        buffer[counter++] = static_cast<uint8_t>((base << 3 | delTag) << 3 | subTag);

        //
        // 2. Byte
        if (b.DeletionQV() < 0 || b.DeletionQV() > 15) 
            throw PBException("DeletionQV OOB");

        if (b.SubstitutionQV() < 0 || b.SubstitutionQV() > 15) 
            throw PBException("SubstitutionQV OOB");

        buffer[counter++] = static_cast<uint8_t>(b.DeletionQV() << 4 | b.SubstitutionQV());

        //
        // 3. Byte
        if (b.InsertionQV() < 0 || b.InsertionQV() > 15) 
            throw PBException("InsertionQV OOB");

        if (p.MergeQV() < 0 || p.MergeQV() > 15) 
            throw PBException("MergeQV OOB");

        buffer[counter++] = static_cast<uint8_t>(b.InsertionQV() << 4 | p.MergeQV());

        //
        // 4. Byte
        if (p.AltLabelQV() < 0 || p.AltLabelQV() > 15) 
            throw PBException("AltLabelQV OOB");

        if (p.LabelQV() < 0 || p.LabelQV() > 15) 
            throw PBException("LabelQV OOB");

        buffer[counter++] = static_cast<uint8_t>(p.AltLabelQV() << 4 | p.LabelQV());

        //
        // 5. Byte
        const auto altLabel = NucleotideLabelToBaz(p.AltLabel());
        if (altLabel > 5) 
            throw PBException("AltLabel OOB");

        const auto label = NucleotideLabelToBaz(p.Label());
        if (p.Label() == NucleotideLabel::NONE
            || label > 4) 
            throw PBException("Label OOB");
        
        buffer[counter++] = static_cast<uint8_t>(
                            (((isBase << 1
                               | isPulse) << 3)
                               | altLabel) << 3
                               | label);

        //
        // 6. Byte
        // IPD
        if (ipds[i] >= 255) // 32 bit IPD
        {
            numIPDs++;
            buffer[counter++] = 255;
            memcpy(&buffer[counter], &ipds[i], sizeof(uint32_t));
            counter += sizeof(uint32_t);
        }
        else // 8 bit IPD
        {
            buffer[counter++] = static_cast<uint8_t>(ipds[i]);
        }

        //
        // Next Byte
        // PW
        if (p.Width() >= 255) // 32 bit PW
        {
            buffer[counter++] = 255;
            uint32_t tmp = p.Width();
            memcpy(&buffer[counter], &tmp, sizeof(uint32_t));
            counter += sizeof(uint32_t);
        }
        else // 8 bit PW
        {
            buffer[counter++] = static_cast<uint8_t>(p.Width());
        }

        //
        // Next Byte
        // PKMEAN
        const auto meanSignalUnscaled = std::round(p.MeanSignal() * 10);
        const uint16_t meanSignal = static_cast<uint16_t>(meanSignalUnscaled > 65535 ? 65535 : meanSignalUnscaled);
        if (meanSignal >= 255) // 16 bit PKMEAN
        {
            buffer[counter++] = 255;
            uint16_t tmp = static_cast<uint16_t>(meanSignal);
            memcpy(&buffer[counter], &tmp, sizeof(uint16_t));
            counter += sizeof(uint16_t);
        }
        else // 8 bit PKMEAN
        {
            buffer[counter++] = static_cast<uint8_t>(meanSignal);
        }
        
        //
        // Next Byte
        // PKMID
        const auto midSignalUnscaled = std::round(p.MidSignal() * 10);
        const uint16_t midSignal = static_cast<uint16_t>(midSignalUnscaled > 65535 ? 65535 : midSignalUnscaled);
        if (midSignal >= 255) // 16 bit PKMID
        {
            buffer[counter++] = 255;
            uint16_t tmp = static_cast<uint16_t>(midSignal);
            memcpy(&buffer[counter], &tmp, sizeof(uint16_t));
            counter += sizeof(uint16_t);
        }
        else // 8 bit PKMID
        {
            buffer[counter++] = static_cast<uint8_t>(midSignal);
        }

        // //
        // // Next Byte
        // // PKMEAN2
        // if (p.MeanSignal() >= 255) // 16 bit PKMEAN2
        // {
        //     buffer[counter++] = 255;
        //     uint16_t tmp = static_cast<uint16_t>(p.MeanSignal());
        //     memcpy(&buffer[counter], &tmp, sizeof(uint16_t));
        //     counter += sizeof(uint16_t);
        // }
        // else // 8 bit PKMEAN
        // {
        //     buffer[counter++] = p.MeanSignal();
        // }
        
        // //
        // // Next Byte
        // // PKMID2
        // if (p.MidSignal() >= 255) // 16 bit PKMID2
        // {
        //     buffer[counter++] = 255;
        //     uint16_t tmp = static_cast<uint16_t>(p.MidSignal());
        //     memcpy(&buffer[counter], &tmp, sizeof(uint16_t));
        //     counter += sizeof(uint16_t);
        // }
        // else // 8 bit PKMID
        // {
        //     buffer[counter++] = p.MidSignal();
        // }
    }
    if (counter != *packetByteStreamSize)
        throw PBException(
            "SliceToPulsePacket Error " +
            std::to_string(counter) + " " +
            std::to_string(*packetByteStreamSize) + " " +
            std::to_string(numExtensions16bit) + " " +
            std::to_string(numExtensions32bit) + " " +
            std::to_string(numIPDs));
    return buffer;
}

template <typename TMetric>
MemoryBufferView<uint8_t> PrimaryToBaz<TMetric>::SliceToBasePacket(
        const Basecall* basecall,
        const uint16_t numEvents,
        const uint32_t zmwId,
        MemoryBuffer<uint8_t>& allocator,
        uint32_t* packetByteStreamSize,
        uint16_t* numIncorporatedEvents)
{
    std::vector<uint16_t> ipds;
    ipds.reserve(numEvents);

    uint16_t numBases = 0;

    int currentBaseFrame = currentBaseFrames_[zmwId];
    for (size_t i = 0; i < numEvents; ++i)
    {
        const auto& b = basecall[i];
        const auto& p = b.GetPulse();
        if (!b.IsNoCall())
        {
            ipds.emplace_back(p.Start() - currentBaseFrame); 
            currentBaseFrame = p.Start() + p.Width();
            ++numBases;
        }
    }
    currentBaseFrames_[zmwId] = currentBaseFrame;

    constexpr 
    uint32_t packetByteSize = ( 2 // base
                              + 3 // deletionTag
                              + 3 // substitutionTag
                              + 4 // deletionQV
                              + 4 // substitutionQV
                              + 4 // insertionQV
                              + 4 // mergeQV
                              + 8 // IPD
                              )/8;// Bits to bytes
    static_assert(packetByteSize == BazWriter<TMetric>::packetSizeProd,"mismatch in packetSizeProd calc");
    *packetByteStreamSize = packetByteSize * numBases;
    *numIncorporatedEvents = numBases;

    if (numBases == 0) return MemoryBufferView<uint8_t>{};
    
    MemoryBufferView<uint8_t> buffer = allocator.Allocate(*packetByteStreamSize);

    size_t counter = 0;
    for (size_t i = 0, j = 0; i < numEvents; ++i)
    {
        const auto& b = basecall[i];
        const auto& p = b.GetPulse();

        if (b.IsNoCall()) continue;
        
        // 
        // 1. Byte
        if (b.Base() == NucleotideLabel::NONE)
            throw PBException("Base is NONE");
        const uint8_t base = NucleotideLabelToBaz(b.Base());
        if (base > 3) throw PBException("Base OOB");

        const auto delTag = NucleotideLabelToBaz(b.DeletionTag());
        if (b.DeletionTag() == NucleotideLabel::NONE
            || delTag > 4)
            throw PBException("DeletionTag OOB");

        const auto subTag = NucleotideLabelToBaz(b.SubstitutionTag());
        if (b.SubstitutionTag() == NucleotideLabel::NONE
            || subTag > 4) 
            throw PBException("SubstitutionTag OOB");

        buffer[counter++] = static_cast<uint8_t>((base << 3 | delTag) << 3 | subTag);

        // 
        // 2. Byte
        const auto delQV = b.DeletionQV();
        if (delQV < 0 || delQV > 15) 
            throw PBException("DeletionQV OOB");

        const auto subQV = b.SubstitutionQV();
        if (subQV < 0 || subQV > 15) 
            throw PBException("SubstitutionQV OOB");

        buffer[counter++] = static_cast<uint8_t>(delQV << 4 | subQV);

        //
        // 3. Byte
        const auto mergeQV = p.MergeQV();
        if (mergeQV < 0 || mergeQV > 15) 
            throw PBException("MergeQV OOB");

        const auto insQV = b.InsertionQV();
        if (insQV < 0 || insQV > 15) 
            throw PBException("InsertionQV OOB");

        buffer[counter++] = static_cast<uint8_t>(insQV << 4 | mergeQV);

        //
        // 4. Byte
        buffer[counter++] = codec.FrameToCode(ipds[j++]);
    }

    return buffer;
}

template <typename TMetric>
MemoryBufferView<uint8_t> PrimaryToBaz<TMetric>::SliceToBasePacketMinimal(
        const Basecall* basecall,
        const uint16_t numEvents,
        const uint32_t zmwId,
        MemoryBuffer<uint8_t>& allocator,
        uint32_t* packetByteStreamSize,
        uint16_t* numIncorporatedEvents)
{
    uint16_t numBases = 0;

    constexpr 
    uint32_t packetByteSize = ( 2 // base
                              + 4 // overallQV
                              + 2 // zeros
                              + 8 // IPD
                              + 8 // PW
                              )/8;// Bits to bytes
    static_assert(packetByteSize == BazWriter<TMetric>::packetSizeProdMin,"mismatch in packetSizeProdMin");

    int currentBaseFrame = currentBaseFrames_[zmwId];

    // Somewhat over-allocate, in order to avoid computing numPulses separately.
    // External code will still only think this memory segment is numBases long.
    MemoryBufferView<uint8_t> buffer = allocator.Allocate(packetByteSize * numEvents);

    size_t counter = 0;
    for (size_t i = 0; i < numEvents; ++i)
    {
        const auto& b = basecall[i];
        const auto& p = b.GetPulse();

        if (b.IsNoCall()) continue;

        numBases++;

        // 
        // 1. Byte
        if (b.Base() == NucleotideLabel::NONE)
            throw PBException("Base is NONE");
        const uint8_t base = NucleotideLabelToBaz(b.Base());
        if (base > 3) throw PBException("Base OOB");

        int8_t compoundQV = 0;
        // Short circuit this when we have no qv.  Note there is a short
        // circuit inside Quality() as well, but even just invoking that function
        // followed by an immediate return has proven expensive.
        if (b.DeletionQV() || b.InsertionQV() || b.SubstitutionQV())
            compoundQV = b.Quality();
        if (compoundQV < 0 || compoundQV > 15) 
            throw PBException("CompoundQV OOB");

        buffer[counter++] = static_cast<uint8_t>((base << 4 | compoundQV) << 2);

        //
        // 2. Byte
        auto ipd = p.Start() - currentBaseFrame;
        currentBaseFrame = p.Start() + p.Width();
        buffer[counter++] = codec.FrameToCode(
                std::numeric_limits<uint16_t>::max() < ipd
                ? std::numeric_limits<uint16_t>::max()
                : static_cast<uint16_t>(ipd));

        //
        // 3. Byte
        buffer[counter++] = codec.FrameToCode(b.GetPulse().Width());
    }
    currentBaseFrames_[zmwId] = currentBaseFrame;
    *packetByteStreamSize = packetByteSize * numBases;
    *numIncorporatedEvents = numBases;

    return buffer;
}

//
// Explicit instantiations
//

template class PrimaryToBaz<SequelMetricBlock>;
template class PrimaryToBaz<SpiderMetricBlock>;

}}
