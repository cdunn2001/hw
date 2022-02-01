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

// Programmer: John Nguyen, Armin Töpfer

#include <atomic>

namespace PacBio {
namespace Primary {
namespace Postprimary {

namespace internal {

static const std::string augmentSequence = "AAAGGGTTTCC";

std::string GetTagSequence(const uint32_t holeNumber)
{
    // Rotating encoding scheme of previous nucleotide-current base3 digit.
    static const char rotCode[4][3] = {
        {'C', 'G', 'T'},    // A - {0,1,2}
        {'G', 'T', 'A'},    // C - {0,1,2}
        {'T', 'A', 'C'},    // G - {0,1,2}
        {'A', 'C', 'G'}     // T - {0,1,2}
    };

    static const int charToIdx[90] =
        {
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1,  0,  1, // '0', '1'
            2, -1, -1, -1, -1, -1, -1, -1, -1, -1, // '2'
            -1, -1, -1, -1, -1,  0, -1,  1, -1, -1, // 'A', 'C'
            -1,  2, -1, -1, -1, -1, -1, -1, -1, -1, // 'G'
            -1, -1, -1, -1,  3, -1, -1, -1, -1, -1, // 'T'
        };

    // 1. Convert to base 3 and pad left with zeros to 21 digits.
    std::ostringstream oss;
    uint32_t v = holeNumber;
    while (v != 0)
    {
        uint32_t r = v % 3;
        oss << r;
        v /= 3;
    }
    std::string s = oss.str();
    std::reverse(std::begin(s), std::end(s));
    s.insert(s.begin(), 21 - s.size(), '0');
    assert(s.size() == 21);

    // 2. Convert using rotating scheme.
    std::string tagSequence;
    tagSequence.reserve(augmentSequence.size() + s.size());
    tagSequence += augmentSequence;

    char prevBase = 'A';
    for (size_t s_pos = 0; s_pos < s.size(); ++s_pos)
    {
        prevBase = rotCode[charToIdx[static_cast<int>(prevBase)]][charToIdx[static_cast<int>(s[s_pos])]];
        tagSequence += prevBase;
    }
    assert(tagSequence.size() == 32);

    return tagSequence;
}

std::pair<bool,int> CheckZMW(const std::string &sequence, const std::string tagSequence)
{
    // Compare to actual sequence looking for first augment sequence to begin match.
    size_t startMatch = sequence.find(augmentSequence);
    if (startMatch == std::string::npos)
    {
        return std::make_pair(false, -1);
    }
    else
    {
        // Do exact match for now, we could probably get away with a quick checksum of every 32 bases.
        while (startMatch < sequence.size())
        {
            size_t endMatch = std::min(sequence.size() - startMatch, tagSequence.size());
            if (sequence.compare(startMatch, endMatch, tagSequence, 0, endMatch))
            {
                return std::make_pair(false, startMatch);
            }
            startMatch += tagSequence.size();
        }
    }

    return std::make_pair(true, 0);
}

bool CheckZMW(const std::string &sequence, const uint32_t holeNumber)
{
    const std::string tagSequence = GetTagSequence(holeNumber);
    const auto ret = CheckZMW(sequence, tagSequence);
    return ret.first;
}

}

class SentinelVerification
{
public: // structors
    // Default constructor
    SentinelVerification() = default;
    // Move constructor
    SentinelVerification(SentinelVerification&&) = delete;
    // Copy constructor
    SentinelVerification(const SentinelVerification&) = delete;
    // Move assignment operator
    SentinelVerification& operator=(SentinelVerification&&) = delete;
    // Copy assignment operator
    SentinelVerification& operator=(const SentinelVerification&) = delete;
    // Destructor
    ~SentinelVerification() {};

public: // modifying methods
    bool CheckZMW(const std::string &sequence, const uint32_t holeNumber)
    {
        const std::string tagSequence = internal::GetTagSequence(holeNumber);
        const auto retVal = internal::CheckZMW(sequence, tagSequence);
        if (retVal.first == false)
        {
            std::ostringstream oss;
            oss << holeNumber << "," << tagSequence;
            if (retVal.second < 0)
                oss << ",NA,NA";
            else
                oss << "," << retVal.second << "," << sequence.substr(retVal.second, tagSequence.size());
            unmatchedZMWsInfo.push_back(oss.str());
            numSentinelZMWs++;
        }
        else
        {
            numNormalZMWs++;
        }

        return retVal.first;
    }

public: // non-modifying methods
    int NumSentinelZMWs() const
    { return numSentinelZMWs; }

    int NumNormalZMWs() const
    { return numNormalZMWs; }

    const std::vector<std::string>& UnmatchedZMWsInfo() const
    { return unmatchedZMWsInfo; }

private:
    std::vector<std::string> unmatchedZMWsInfo;
    std::atomic_int numSentinelZMWs{0};
    std::atomic_int numNormalZMWs{0};
};

}}}

