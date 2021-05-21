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

#include <algorithm>
#include <assert.h>
#include <stdlib.h>
#include <utility>
#include <vector>

#include <postprimary/alignment/Cell.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Provides helper method for AdapterFinding
class SpacedSelector
{
public:
    /// Finds possible adapter end positions with minimal spacing
    static std::vector<Cell> FindEndPositions(std::vector<Cell>&& lastRow,
                                              uint32_t minSpacing)
    {
        std::vector<SmallCell> sc;
        sc.reserve(lastRow.size());
        for (const auto& x : lastRow)
            sc.emplace_back(x.score, x.score, -1, x.jEndPosition);

        // For each subsequent item, 
        // figure out what the best preceding item would be
        for(size_t i = 1; i < sc.size(); i++)
        {
            auto& iSc = sc[i];
            // Go through previous items, 
            // and find the one that satisfies the spacing constraint,
            // and has the highest score.
            int32_t newScore;
            for (int32_t j = i - 1; j != -1; j--)
            {
                auto& curJJ = sc[j];
                if (iSc.jEndPosition - curJJ.jEndPosition > minSpacing)
                {
                    newScore = curJJ.newScore + iSc.score;
                    if (newScore > iSc.newScore)
                    {
                        iSc.newScore = newScore;
                        iSc.prevItem = j;
                    }
                }
            }
        }

        std::vector<int32_t> score;
        score.reserve(lastRow.size());
        for (const auto& x : sc)
            score.emplace_back(x.newScore);

        // Find the cell with the highest total score, 
        // then trace back through the included items.
        auto result = std::max_element(score.begin(), score.end());
        auto end = std::distance(score.begin(), result);
        std::vector<Cell> final;

        while (end >= 0)
        {
            final.push_back(std::move(lastRow[end]));
            end = sc[end].prevItem;
        }
        std::reverse(final.begin(), final.end());
        return final;
    }

public:
    SpacedSelector() = delete;
    // Move constructor
    SpacedSelector(SpacedSelector&&) = delete;
    // Copy constructor is deleted!
    SpacedSelector(const SpacedSelector&) = delete;
    // Move assignment operator
    SpacedSelector& operator=(SpacedSelector&&) = delete;
    // Copy assignment operator is deleted!
    SpacedSelector& operator=(const SpacedSelector&) = delete;
    // Destructor
    ~SpacedSelector() = delete;

private:
    struct SmallCell
    {
        int32_t  score;
        int32_t  newScore;
        int32_t  prevItem;
        uint32_t jEndPosition;

        SmallCell(int32_t scoreArg, int32_t newScoreArg, int32_t prevItemArg,
                  uint32_t jEndPositionArg)
            : score(scoreArg)
            , newScore(newScoreArg)
            , prevItem(prevItemArg)
            , jEndPosition(jEndPositionArg)
        {
        }
    };

};

}}} // ::PacBio::Primary::Postprimary


