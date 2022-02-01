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

#include <fstream>

#include <pacbio/logging/Logger.h>

#include "StaticHQRegionFinder.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

StaticHQRegionFinder::StaticHQRegionFinder(const std::string& fakeHQFile)
      : HQRegionFinder()
{
    if (!fakeHQFile.empty())
    {
        // Parse provided HQ regions
        std::ifstream file(fakeHQFile);
        std::string str; 
        while (std::getline(file, str))
        {
            if (str[0] != '>') continue;
            std::vector<std::string> elems;
            std::stringstream ss(str);
            std::string item;
            while (std::getline(ss, item, '/'))
                elems.push_back(item);

            std::vector<int> regions;
            std::stringstream ssInterval(elems[2]);
            while (std::getline(ssInterval, item, '_'))
                regions.push_back(std::stoi(item));

            fakeHQRegions_.emplace_back(regions[0], regions[1]);
        }
    }
}

std::pair<size_t, size_t> StaticHQRegionFinder::FindHQRegion(const BlockLevelMetrics& ,
                                                             const EventData& zmw) const
{
    if (fakeHQRegions_.size() > 0)
    {
        try
        {
            // This is in bases coordinates (pre exclusions) unfortunately
            auto fakeHQ = fakeHQRegions_.at(zmw.ZmwIndex());
            // The input base coordinates don't account for pulse exclusions
            // that may have happened.  Need to account for them.
            auto BazBaseToPulse = [&](size_t idx)
            {
                const auto& isBase = zmw.IsBase();
                const auto& excludedBases = zmw.ExcludedBases();
                size_t origBaseCount = 0;
                for (size_t pulseIdx = 0; pulseIdx < isBase.size(); ++pulseIdx)
                {
                    if (isBase[pulseIdx] || excludedBases[pulseIdx])
                    {
                        origBaseCount++;
                    }
                    if (origBaseCount == idx+1)
                        return pulseIdx;
                }
                if (origBaseCount != idx)
                    throw PBException("Error converting base to pulse coordinates");
                return idx;
            };
            fakeHQ.first = BazBaseToPulse(fakeHQ.first);
            fakeHQ.second = BazBaseToPulse(fakeHQ.second);
            return fakeHQ;
        } catch (std::exception& e)
        {
            PBLOG_ERROR << "zmw index " << zmw.ZmwIndex() << " is out of range for the provided fake hq file!";
            throw e;
        }
    } else {
        return {0, zmw.NumBases()};
    }
}

}}} // ::PacBio::Primary::Postprimary



