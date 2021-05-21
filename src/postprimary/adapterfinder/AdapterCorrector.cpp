// Copyright (c) 2018-2020, Pacific Biosciences of California, Inc.
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

#include <unordered_map>
#include <vector>
#include <numeric>

#include <pacbio/logging/Logger.h>

#include <boost/numeric/conversion/cast.hpp>

#include <bazio/SequenceUtilities.h>

#include <postprimary/alignment/PairwiseAligner.h>

#include "AdapterCorrector.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

constexpr float SensitiveAdapterLabeler::minSoftAccuracy;
constexpr float SensitiveAdapterLabeler::minHardAccuracy;
constexpr int   SensitiveAdapterLabeler::minFlankingScore;
constexpr bool  SensitiveAdapterLabeler::localAlnFlanking;
constexpr int   SensitiveAdapterLabeler::flankLength;

// These are algorithm constraints, and therefore not configurable
constexpr int minInsertSizeForAC = 100;
constexpr float maxShortInsertFractionForAC = 0.25;

bool shortInsertPresent(const std::vector<int>& sizes,
                        int threshold=minInsertSizeForAC)
{
    int numShort = 0;
    for (const auto& sz : sizes)
    {
        if (sz < threshold)
            ++numShort;
    }
    if (numShort > sizes.size() * maxShortInsertFractionForAC)
        return true;
    return false;
}

template <typename T>
T modifyingQuantile(std::vector<T>& data,
                    float quantile)
{
    size_t quanti = data.size() * quantile;
    std::nth_element(data.begin(), data.begin() + quanti, data.end());
    return data[quanti];
}

std::vector<int> PickCandidates(const std::vector<int> candidates,
                                int expectedSpacing)
{
    // Use a simple dynamic programming algorithm to pick the set of candidates
    // spaced most closely to the expectedSpacing
    std::vector<int> scores;
    std::vector<int> moves;
    int maxPossibleScore = expectedSpacing;
    // We want to include a move from the end as a no-op alternative to some
    // random point in the last partial subread, which might be better to
    // accept than skip purely because it is beyond a certain distance.
    for (size_t i = 0; i < candidates.size(); ++i)
    {
        // Calculate the "transition to start" score:
        // Start by assuming this adapter is great, the transition score is the
        // highest possible
        int maxScore = maxPossibleScore;
        int adpLocation = candidates[i];
        int distFromStart = adpLocation;
        // But if the distance between this adapter and the beginning is
        // greater than expected spacing, downrate the score by the difference
        if (distFromStart > expectedSpacing)
            maxScore -= (distFromStart - maxScore);

        // Lets look at all previous adapters, to see if we can do better than
        // the -1 move calculated above (which marks the current adapter as the
        // last one in the chain to accept).
        int move = -1;
        for (size_t j = 0; j < i; ++j)
        {
            // Calculate the transition score from this node:
            int dist = adpLocation - candidates[j];
            int score = maxPossibleScore;
            if (dist > maxPossibleScore)
                score -= 2 * (dist - maxPossibleScore);
            else
                score -= 3 * (maxPossibleScore - dist);
            if (scores[j] + score > maxScore)
            {
                maxScore = scores[j] + score;
                move = j;
            }
        }
        scores.push_back(maxScore);
        moves.push_back(move);
    }

    // Find the best adapter to start at:
    int start = -1;
    int maxScore = 0;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > maxScore)
        {
            maxScore = scores[i];
            start = i;
        }
    }

    // Backtrack to find the list of optimal adapters:
    std::vector<int> best;
    int i = start;
    while (i >= 0)
    {
        if (i < static_cast<int>(candidates.size()))
            best.push_back(i);
        i = moves[i];
    }
    std::reverse(best.begin(), best.end());
    return best;
}

std::vector<RegionLabel> RemoveFalsePositives(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqregion)
{
    // We can't use partial subreads for this application
    if (adapters.size() == 0)
        return adapters;
    if (adapters.size() < 3)
        return PalindromeSearch(bases, adapters, hqregion, true);

    // Find the median
    std::vector<int> sizes;
    sizes.reserve(adapters.size());
    for (size_t i = 1; i < adapters.size(); ++i)
    {
        // look for the gap between adapters
        sizes.push_back(adapters[i].begin - adapters[i - 1].end);
    }
    // The "PickCandidates" algorithm doesn't work with non-standard loop
    // topologies. A population of small inserts is a common phenotype
    if (shortInsertPresent(sizes))
        return adapters;

    // If we have a bunch of tiny false positives, we don't want them pulling
    // down the median.
    int medianLength = modifyingQuantile(sizes, 0.5);

    std::vector<int> distsFromStart;
    for (const auto& adp : adapters)
        distsFromStart.push_back(adp.begin - hqregion.begin);
    auto best = PickCandidates(distsFromStart, medianLength);
    std::vector<RegionLabel> ret;
    for (int adpi : best)
        ret.push_back(adapters[adpi]);
    return ret;
}

std::vector<int> FlagSubreadLengthOutliers(
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqRegion,
    float lengthThreshold)
{
    std::vector<int> flagged;
    // We can't use partial subreads for this application
    if (adapters.size() < 2)
        return flagged;

    // Find the median
    std::vector<int> sizes;
    sizes.reserve(adapters.size());
    // look for the gap between adapters
    std::transform(std::next(adapters.begin()), adapters.end(),
                   adapters.begin(),
                   std::back_inserter(sizes),
                   [] (const auto& adp1, const auto& adp2)
                   { return adp1.begin - adp2.end; });

    auto origSizes = sizes;
    // Bias a little high, in case we have 50% short inserts
    double averageLen = modifyingQuantile(sizes, 0.6);
    // If everything is short, bail!
    if (averageLen < minInsertSizeForAC)
        return flagged;
    // If some but not all are short, we have a bimodal situation. Normally we
    // would bail, but lets flag these subreads and look for RC adapter
    // sequences instead
    bool bimodal = shortInsertPresent(sizes, std::max(minInsertSizeForAC, static_cast<int>(0.75 * averageLen)));

    // This doesn't have to be perfect, and we already have our (slightly
    // biased) average, so we'll use it
    double variance = 0;
    for (const auto& val : sizes)
        variance += (averageLen - val) * (averageLen - val);
    variance /= sizes.size();
    double stdDev = sqrt(variance);

    // Flag reads that are ~2X larger than the median, but since we're not
    // addressing the first or last read here we need to map the size index
    // to the subread index (+1).
    // We also flag subreads if the standard deviation of insert sizes exceeds
    // a threshold and the insert size is simply greater than the median.
    for (size_t i = 0; i < origSizes.size(); ++i)
    {
        if (bimodal
            || ((origSizes[i] / averageLen) > lengthThreshold)
            || (stdDev > 0.4 * averageLen && origSizes[i] > (averageLen * 1.4)))
        {
            flagged.push_back(i + 1);
        }
    }
    // Flag the start and end reads if they are larger than the median
    if (bimodal || (adapters[0].begin - hqRegion.begin) / averageLen > 1.1)
        flagged.insert(flagged.begin(), 0);
    if (bimodal || (hqRegion.end - adapters.back().end) / averageLen > 1.1)
        flagged.push_back(adapters.size());

    return flagged;
}

std::vector<int> FlagAbsentAdapters(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const ScoringScheme& scoreScheme,
    int flankSize,
    int flankScore)
{
    std::vector<int> flagged;
    if (adapters.size() < 2)
        return flagged;

    // Find the median
    std::vector<int> sizes;
    sizes.reserve(adapters.size());
    for (size_t i = 1; i < adapters.size(); ++i)
    {
        // look for the gap between adapters
        sizes.push_back(adapters[i].begin - adapters[i - 1].end);
    }
    if (shortInsertPresent(sizes))
        return flagged;

    int nBases = boost::numeric_cast<int>(bases.size());
    int nAdapters = boost::numeric_cast<int>(adapters.size());

    std::string lastAdapter = bases.substr(
        std::max(0, adapters[0].begin - flankSize),
        std::min(nBases, adapters[0].end - adapters[0].begin + 2 * flankSize));

    // compare each adapter flank to the last, flag those subreads that
    // match too well. This of course will break inserts that are perfect
    // panlindromes
    for (int i = 1; i < nAdapters; ++i)
    {
        if (adapters[i].begin - adapters[i-1].end < (2 * flankSize))
            continue;
        std::string thisAdapter = bases.substr(
            adapters[i].begin - flankSize,
            std::min(nBases,
                     adapters[i].end - adapters[i].begin + 2 * flankSize));
        auto maxscore = PairwiseAligner::SWComputeMaxScore(
            lastAdapter,
            thisAdapter,
            scoreScheme);
        if (maxscore > flankScore)
        {
            // yes we're comparing this adapter to the one before, but we're
            // flagging the subread between them
            flagged.push_back(i);
        }
        lastAdapter = thisAdapter;
    }
    // We want to reanalyze the first and last subreads if we're getting
    // consistent hits (they won't have the adapter either).
    // This is really tricky when you've already done missed adapter correction
    // though, because you might fix all but one internal subread and have one
    // internal and the two flanking subreads to fix
    //
    // This is also really tricky because low accuracy regions around adapters
    // can produce flag false negatives, preventing those adapters from ever
    // being fixed. It is much harder to get a false positive flag, but even
    // then it will probably only cost analysis time, as the corrector false
    // positive rate is separate and also low.
    //
    // The compromise for now is to flag every subread if a missing adapter is
    // suggested. We can roll this back to just the flanks if we decide that
    // costs too much, but we will miss inner missing adapters as a result
    if (flagged.size() > 0)
    {
        // add the flanks:
        //flagged.insert(flagged.begin(), 0);
        //flagged.push_back(adapters.size());
        // add every subread:
        flagged.resize(adapters.size() + 1);
        std::iota(flagged.begin(), flagged.end(), 0);
    }
    return flagged;
}

std::vector<RegionLabel> SensitiveAdapterSearch(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqRegion,
    const std::vector<int>& flagged,
    const AdapterLabeler& sensitiveAdapterLabeler,
    const AdapterLabeler& rcAdapterLabeler)
{
    bool lookForRC = true;
    if (flagged.size() == 0)
        return adapters;

    bool stemFound = false;
    for (const auto& adp : adapters)
        if (adp.hasStem)
            stemFound = true;

    bool rcFound = false;
    std::vector<RegionLabel> newAdapters;
    newAdapters.reserve(adapters.size() + flagged.size());
    auto curFlag = flagged.begin();
    // split flagged subreads if possible:
    for (size_t i = 0; i <= adapters.size(); ++i)
    {
        if (rcFound || (curFlag != flagged.end() && *curFlag == i))
        {
            RegionLabel subread;
            if (i == 0)
            {
                subread.begin = hqRegion.begin;
                if (adapters.size() > 0)
                    subread.end = adapters[i].begin;
                else
                    subread.end = hqRegion.end;
            } else if (i == adapters.size()) {
                subread.begin = adapters[i - 1].end;
                subread.end = hqRegion.end;
            } else {
                subread.begin = adapters[i - 1].end;
                subread.end = adapters[i].begin;
            }

            // We'll use existing infrastructure (configured for increased
            // sensitivity) to identify additional reads

            // By this point we know whether or not we've seen the stem sequence
            // in this ZMW. We're never going to look to extend the stem
            // sequence. We just need to set the stem length to the right size
            // for this ZMW (where the right size is the number of bases to
            // exclude, 0 for TC6, 8 for TC6-loop-only

            // This is a little confusing. We're using stem length here to refer
            // to the number of bases on each end of the adapter that should be
            // ignored if "lookForLoop/trimToLoop" is off. If we found the stem
            // sequence in other adapters, we set stem length to 0 (lookForLoop
            // and trimToLoop are off in sensitiveAdapterLabeler), and we end up
            // looking for the entire adapter sequence.
            const auto newHits = stemFound ?
                sensitiveAdapterLabeler.Label(bases, subread, false, 0) :
                sensitiveAdapterLabeler.Label(bases, subread, false);

            // Pick the best new hit:
            if (newHits.size() > 1)
            {
                int best = 0;
                for (size_t ii = 0; ii < newHits.size(); ++ii)
                {
                    float accdelta = newHits[ii].accuracy
                                     - newHits[best].accuracy;
                    int flankdelta = newHits[ii].flankingScore
                                     - newHits[best].flankingScore;
                    if (accdelta > 0 && flankdelta > 0)
                        best = ii;
                    else if (accdelta > 0.1 || flankdelta > 10)
                        best = ii;
                    // else it is worse and could be a false positive!
                }
                newAdapters.push_back(newHits[best]);
            }
            else if (newHits.size() == 1)
            {
                newAdapters.push_back(newHits[0]);
            }

            if (lookForRC)
            {
                // Now lets look for the reverse complement of our adapter sequence
                // and see what we find!
                const auto rcHits = stemFound ?
                    rcAdapterLabeler.Label(bases, subread, false, 0) :
                    rcAdapterLabeler.Label(bases, subread, false);
                if (rcHits.size() > 0)
                    rcFound = true;

                // Pick the best new hit:
                if (rcHits.size() > 1)
                {
                    int best = 0;
                    for (size_t ii = 0; ii < rcHits.size(); ++ii)
                    {
                        float accdelta = rcHits[ii].accuracy
                                         - rcHits[best].accuracy;
                        int flankdelta = rcHits[ii].flankingScore
                                         - rcHits[best].flankingScore;
                        if (accdelta > 0 && flankdelta > 0)
                            best = ii;
                        else if (accdelta > 0.1 || flankdelta > 10)
                            best = ii;
                        // else it is worse and could be a false positive!
                    }
                    newAdapters.push_back(rcHits[best]);
                }
                else if (rcHits.size() == 1)
                {
                    newAdapters.push_back(rcHits[0]);
                }
            }

            // bookkeeping for the next round
            ++curFlag;
        }
        if (i < adapters.size()) newAdapters.push_back(adapters[i]);
    }
    return newAdapters;
}

int diff(int val0, int val1)
{
    return std::abs(val0 - val1);
}

size_t diff(size_t val0, size_t val1)
{
    return std::max(val0, val1) - std::min(val0, val1);
}

std::tuple<std::string, std::vector<int>, std::vector<RegionLabel>>
CompressHomopolymers(const std::string& bases,
                     const std::vector<RegionLabel>& adapters)
{
    std::string compressed;
    compressed.reserve(bases.size());
    std::vector<int> origIndex;
    origIndex.reserve(bases.size());
    std::vector<RegionLabel> shiftedAdapters;
    shiftedAdapters.reserve(adapters.size());
    char last = 'z'; // perhaps a better default is in order?
    int adpStart = 0;
    size_t adapteri = 0;
    for (size_t i = 0; i < bases.size(); ++i)
    {
        if (bases[i] != last)
        {
            last = bases[i];
            compressed += bases[i];
            origIndex.push_back(i);
        }
        if (adapteri < adapters.size())
        {
            if (i == adapters[adapteri].begin)
            {
                adpStart = compressed.size();
            }
            if (i == adapters[adapteri].end)
            {
                shiftedAdapters.emplace_back(
                        adpStart,
                        compressed.size(),
                        adapters[adapteri].score,
                        adapters[adapteri].sequenceId,
                        adapters[adapteri].type);
                ++adapteri;
            }
        }
    }
    if (adapteri < adapters.size())
    {
        shiftedAdapters.emplace_back(
                adpStart,
                compressed.size(),
                adapters[adapteri].score,
                adapters[adapteri].sequenceId,
                adapters[adapteri].type);
    }
    return std::make_tuple(compressed, origIndex, shiftedAdapters);
}

std::vector<RegionLabel> PalindromeSearch(
    const std::string& bases,
    const std::vector<RegionLabel>& adapters,
    const RegionLabel& hqRegion,
    bool removeOnly)
{
    // Don't go above 15 while using int32 for hash indices
    int kmerSize = 15;
    int centerSampleSize = 50;
    const int nAdapters = boost::numeric_cast<int>(adapters.size());
    bool homopolymerCompression = false;

    std::vector<int> sizes;
    sizes.reserve(adapters.size());
    for (size_t i = 1; i < adapters.size(); ++i)
        sizes.push_back(adapters[i].begin - adapters[i - 1].end);
    int roughInsertSize = modifyingQuantile(sizes, 0.75);
    // Refuse to operate on short inserts
    if (roughInsertSize < 140)
        return adapters;

    std::vector<int> centers;
    if (homopolymerCompression)
    {
        std::vector<int> origIndex;
        std::string insert;
        std::vector<RegionLabel> shiftedAdapters;
        std::tie(insert, origIndex, shiftedAdapters) = CompressHomopolymers(
                bases.substr(hqRegion.begin,
                             hqRegion.end - hqRegion.begin),
                adapters);
        centers = PalindromeTrieCenter(
            insert, shiftedAdapters, hqRegion.begin, kmerSize, centerSampleSize);
        // homopolymer compression threw off our indices, we need to
        // "uncompress" the hit locations
        for (auto& center : centers)
        {
            center = origIndex[center - hqRegion.begin]
                     + hqRegion.begin;
        }
    }
    else
    {
        const auto& insert = bases.substr(hqRegion.begin,
                              hqRegion.end - hqRegion.begin);
        centers = PalindromeTrieCenter(
            insert, adapters, hqRegion.begin, kmerSize, centerSampleSize);
    }

    std::vector<RegionLabel> newAdapters;
    newAdapters.reserve(nAdapters);
    bool firstKept = false;
    int adpI = 0;
    for (size_t centerI = 0; centerI < centers.size(); ++centerI)
    {
        auto center = centers[centerI];
        roughInsertSize = (hqRegion.end - hqRegion.begin)/centers.size();
        bool existing = false;
        if (newAdapters.size() > 0
                && center < newAdapters.back().end)
        {
            continue;
        }
        for (int i = adpI; i < nAdapters; ++i)
        {
            const auto& adp = adapters[i];
            int adpLoc = (adp.end + adp.begin)/2;
            // If the next center exists, and is closer to the adapter you're
            // considering, you're already too far into the existing adapter
            // list.
            if (centerI + 1 < centers.size() && diff(center, adpLoc)
                    > diff(centers[centerI + 1], adpLoc))
            {
                // We really don't want overlapping adapters...
                if (center > adp.begin && center < adp.end)
                {
                    existing = true;
                }
                break;
            }
            // if we are close to an existing adapter, accept it and increment
            // our search
            if (diff(center, adpLoc) < std::max(250, roughInsertSize / 10))
            {
                newAdapters.push_back(adp);
                if (i == 0)
                    firstKept = true;
                adpI = i + 1;
                existing = true;
            }
            // If we are not close to the current adapter, we won't get any
            // closer with a future center, might as well increment adpI. But
            // lets check to make sure that our rejection is plausible...
            else
            {
                if (adp.accuracy > 0.85 && (adp.flankingScore == 0 || adp.flankingScore > 125))
                {
                    newAdapters.push_back(adp);
                    if (i == 0)
                        firstKept = true;
                }
                adpI = i + 1;
            }
        }
        if (!removeOnly && !existing && center > hqRegion.begin
                && center < hqRegion.end
                && (newAdapters.size() == 0
                    || center > newAdapters.back().end))
        {
            newAdapters.emplace_back(
                center,
                center,
                0, 0, RegionLabelType::ADAPTER);
            newAdapters.back().accuracy = 0;
            newAdapters.back().flankingScore = 0;
        }
    }
    // Rescue the last adapter if necessary
    if (nAdapters > 0 && adpI < nAdapters
            && hqRegion.end - adapters.back().end < 200
            && (newAdapters.size() == 0
                || adapters.back().begin > newAdapters.back().end))
    {
        newAdapters.push_back(adapters.back());
    }
    // Rescue the first adapter if necessary
    if (nAdapters > 0 && !firstKept
            && adapters.front().begin - hqRegion.begin < 700
            && (newAdapters.size() == 0
                || adapters.front().begin < newAdapters.front().begin))
    {
        newAdapters.insert(newAdapters.begin(), adapters.front());
    }
    for (size_t i = 0; i < newAdapters.size(); ++i)
    {
        const auto& adapter = newAdapters[i];
        for (size_t j = 0; j < i; ++j)
        {
            const auto& other = newAdapters[j];
            if (!(adapter.begin >= other.end || adapter.end <= other.begin))
            {
                PBLOG_ERROR << "Overlapping adapters identified!";
                return adapters;
            }
        }
        if (i > 0)
        {
            // Assert that everything is sorted
            assert(newAdapters[i].begin >= newAdapters[i - 1].begin);
            assert(newAdapters[i].end >= newAdapters[i - 1].end);
            assert(newAdapters[i].begin >= newAdapters[i - 1].end);
        }
    }
    return newAdapters;
}


std::pair<int, int> HashKmer(const std::string& bases,
                             int pos, int kmerSize,
                             const std::vector<int>& powlut)
{
    assert((pos + kmerSize) <= boost::numeric_cast<int>(bases.size()));
    int fdIndex = 0;
    int rcIndex = 0;
    int numAnalogs = 4;
    // In the imaginary perfect suffix tree of all possible kmers that
    // whose leaf array this hash is indexing into, the bases are
    // sorted ACGT, with index offsets of 0, 1, 2, 3, respectively.
    //
    // For example, a perfect trie would look like this for kmers of size 2
    // (with leaf indices containing important metadata listed below)
    //                            0
    //          A           C           G           T
    //  A  C  G  T    A  C  G  T     A  C  G  T    A  C  G  T
    //  0  1  2  3    4  5  6  7     8  9 10 11   12 13 14 15
    //
    // The sequence CGT has two possible kmers that correspond to the
    // following leaves:
    //                      ^                 ^
    // CGT:                 CG                GT
    //
    //  CG == 6  == 1x^1 + 2x^0
    //  GT == 11 == 2x^1 + 3x^0
    //
    // Note: x=4, the number of analogs (and children for each node)
    //
    // And the reverse complement has the following positions:
    //     ^                ^
    // ACG:AC               CG
    //
    // Because this is the RC, we observe CG first, followed by AC:
    //  CG == 6 == 1x^1 + 2x^0
    //  AC == 1 == 0x^1 + 1x^0
    for (int offset = 0; offset < kmerSize; ++offset)
    {
        switch (bases[pos + offset])
        {
            case 'A':
                fdIndex *= numAnalogs;
                rcIndex += powlut[offset] * 3;
                break;
            case 'C':
                fdIndex *= numAnalogs;
                fdIndex += 1;
                rcIndex += powlut[offset] * 2;
                break;
            case 'G':
                fdIndex *= numAnalogs;
                fdIndex += 2;
                rcIndex += powlut[offset] * 1;
                break;
            case 'T':
                fdIndex *= numAnalogs;
                fdIndex += 3;
                break;
        }
    }
    return std::make_pair(fdIndex, rcIndex);
}

std::pair<int, int> UpdateKmerHash(const std::string& bases,
                                   int pos, int kmerSize,
                                   const std::vector<int>& powlut,
                                   int fdIndex, int rcIndex)
{
    assert((pos + kmerSize - 1) < boost::numeric_cast<int>(bases.size()));
    const char oldbase = bases[pos - 1];
    int numAnalogs = 4;
    // In the imaginary perfect suffix tree of all possible kmers that
    // whose leaf array this hash is indexing into, the bases are
    // sorted ACGT, with index offsets of 0, 1, 2, 3, respectively.
    //
    // For example, a perfect trie would look like this for kmers of size 2
    // (with leaf indices containing important metadata listed below)
    //                            0
    //          A           C           G           T
    //  A  C  G  T    A  C  G  T     A  C  G  T    A  C  G  T
    //  0  1  2  3    4  5  6  7     8  9 10 11   12 13 14 15
    //
    // We have an initial hash from HashKmer() of position 0 of CGT:
    //
    //  CG == 6 == 1x^1 + 2x^0
    //
    // Note: x=4, the number of analogs (and children for each node)
    //
    // We want to update the hash so we don't have to create a new one. We
    // subtract the C, promote the G up the tree and add the T leaf:
    //
    //  GT == 11 == (1x^1 + 2x^0 - 1x^1)x + 3x^0
    //
    // We also want to update the reverse complement at the same time, which is
    // similar (note that we observe the CG rckmer vefore the AC rckmer):
    // subtract the G, demote the C, rebase on the new A:
    //
    //  CG == 6 == 1x^1 + 2x^0
    //  AC == 1 == (1x^1 + 2x^0 - 2x^0)/x + 0x^1
    switch (oldbase)
    {
        case 'A':
            fdIndex *= numAnalogs;
            rcIndex -= 3;
            rcIndex /= numAnalogs;
            break;
        case 'C':
            fdIndex -= powlut[kmerSize - 1] * 1;
            fdIndex *= numAnalogs;
            rcIndex -= 2;
            rcIndex /= numAnalogs;
            break;
        case 'G':
            fdIndex -= powlut[kmerSize - 1] * 2;
            fdIndex *= numAnalogs;
            rcIndex -= 1;
            rcIndex /= numAnalogs;
            break;
        case 'T':
            fdIndex -= powlut[kmerSize - 1] * 3;
            fdIndex *= numAnalogs;
            rcIndex /= numAnalogs;
            break;
    }
    const char newbase = bases[pos + kmerSize - 1];
    switch (newbase)
    {
        case 'A':
            rcIndex += powlut[kmerSize - 1] * 3;
            break;
        case 'C':
            fdIndex += 1;
            rcIndex += powlut[kmerSize - 1] * 2;
            break;
        case 'G':
            fdIndex += 2;
            rcIndex += powlut[kmerSize - 1] * 1;
            break;
        case 'T':
            fdIndex += 3;
            break;
    }
    return std::make_pair(fdIndex, rcIndex);
}

namespace {

struct KmerHit
{
    KmerHit(int reverseIndex, int forwardIndex)
        : reverse(reverseIndex)
        , forward(forwardIndex)
    { assert(reverseIndex > forwardIndex); };

    int reverse;
    int forward;
    int Location() const { return (reverse + forward)/2; };
    int Span() const { return reverse - forward; };
};

struct InsertSizeEstimate
{
    int size;
    int minEst;
    int maxEst;
    bool fwHitMethodRejected;
    bool successful;
};

struct Candidate
{
    Candidate(int canDensity, int canLocation)
        : density(canDensity)
        , location(canLocation) {};
    int density;
    int location;
};

double EstimateAccuracy(const std::vector<KmerHit>& rcHits,
                        const std::vector<KmerHit>& forwardHits)
{
    double expectedAccuracy = 0.86;
    int meanDistanceBetweenHits = 0;
    for (size_t i = 1; i < rcHits.size(); ++i)
    {
        meanDistanceBetweenHits += rcHits[i].reverse - rcHits[i - 1].reverse;
    }

    for (size_t i = 1; i < forwardHits.size(); ++i)
    {
        meanDistanceBetweenHits += forwardHits[i].reverse
                                   - forwardHits[i - 1].reverse;
    }
    if (rcHits.size() + forwardHits.size() > 0)
    {
        meanDistanceBetweenHits /= rcHits.size() + forwardHits.size();

        // A lightly calibrated estimate:
        expectedAccuracy = 0.82 + 0.06
            * std::max(0.0, (1.0 - meanDistanceBetweenHits/200.0));
    }
    return expectedAccuracy;
}

InsertSizeEstimate EstimateInsertSize(const std::vector<KmerHit>& rcHits,
                                      const std::vector<KmerHit>& forwardHits,
                                      const std::vector<RegionLabel>& adapters,
                                      const std::string& bases,
                                      double expFwHits,
                                      int hqrOffset)
{
    InsertSizeEstimate estIns;
    estIns.size = bases.size()/2;
    estIns.minEst = bases.size()/2;
    estIns.maxEst = 0;
    estIns.fwHitMethodRejected = false;
    const int nRcHits = boost::numeric_cast<int>(rcHits.size());
    const int nFwHits = boost::numeric_cast<int>(forwardHits.size());
    const int nBases = boost::numeric_cast<int>(bases.size());

    // For an infinite read length with no kmer redundancy, each base will
    // participate in two RC kmers and one forward kmer. In short reads, with
    // say 3x insert passes, bases on the first and last pass will participate
    // in one RC kmer, and only the first and last passes will participate in
    // forward kmers. Therefore you get 3x RC kmers and 1x forward kmers. We'll
    // give a bit of a margin:
    if (nFwHits > nRcHits/4 && nFwHits > expFwHits)
    {
        std::vector<int> fwLengths;
        for (const auto& fwhit : forwardHits)
        {
            fwLengths.push_back(fwhit.Span());
        }
        std::sort(fwLengths.begin(), fwLengths.end());

        // This is 10% of the insert (each fwLength should span two passes):
        int flankSize = fwLengths[fwLengths.size()/2] / 20;
        auto wBegin = fwLengths.begin();
        auto wEnd = fwLengths.begin();
        int peakDensity = 0;
        int peakLength = 0;
        for (size_t i = 0; i < fwLengths.size(); ++i)
        {
            while (wBegin != fwLengths.end()
                    && *wBegin + flankSize < fwLengths[i])
                ++wBegin;
            while (wEnd != fwLengths.end() && fwLengths[i] + flankSize > *wEnd)
                ++wEnd;
            int density = wEnd - wBegin;
            if (density >= peakDensity)
            {
                peakDensity = density;
                peakLength = fwLengths[i];
            }
        }
        int fwMode = peakLength;
        estIns.size = fwMode/2;
        estIns.minEst = std::min(estIns.size, estIns.minEst);
        estIns.maxEst = std::max(estIns.size, estIns.maxEst);

    }
    if (adapters.size() > 0)
    {
        int roughInsertSize = nBases/(adapters.size() + 1);
        if (adapters.size() > 1)
        {
            // Find the median
            std::vector<int> sizes;
            sizes.reserve(adapters.size());
            for (size_t i = 1; i < adapters.size(); ++i)
            {
                // look for the gap between adapters
                if (adapters[i].begin - adapters[i - 1].end > 25)
                {
                    sizes.push_back(adapters[i].begin - adapters[i - 1].end);
                }
            }

            if (sizes.size() > 0)
            {
                // Find the mean:
                int mean = std::accumulate(
                    sizes.begin(), sizes.end(), 0) / sizes.size();

                roughInsertSize = modifyingQuantile(sizes, 0.5);
                estIns.minEst = std::min({mean, estIns.minEst,
                                             roughInsertSize});
                estIns.maxEst = std::max({mean, estIns.maxEst,
                                             roughInsertSize});

                // If there is a huge difference between the min and max
                // estimates (which come from mean, median of adapter based
                // estimation and the smoothed mode of forward hit lengths),
                // and the adapter median is the reason (e.g. because of
                // adapter dimers) use the forward hit length mode
                if (estIns.minEst < estIns.maxEst/5
                        && roughInsertSize < estIns.size)
                {
                    // our estimates vary by a lot, bail on this estimation
                    // method
                    roughInsertSize = estIns.size;
                }
            }
            else
            {
                // Something has gone terribly wrong, with plenty of adapters
                // but no long inserts found:
                estIns.successful = false;
                return estIns;
            }
        }
        else if (adapters.size() == 1)
        {
            roughInsertSize = std::max(
                adapters[0].begin - hqrOffset,
                nBases + hqrOffset - adapters[0].end);
            estIns.minEst = std::min(roughInsertSize, estIns.minEst);
            estIns.maxEst = std::max(roughInsertSize, estIns.maxEst);
        }
        else
        {
            roughInsertSize = nBases;
        }

        if (estIns.size < roughInsertSize / 2.5
                || estIns.size > roughInsertSize * 2.5
                || !(nFwHits > nRcHits/4
                        && nFwHits > expFwHits))
        {
            estIns.size = roughInsertSize;
            estIns.fwHitMethodRejected = true;
        }
    }
    estIns.successful = true;
    return estIns;
}

std::vector<Candidate> SmoothHitDensities(const std::vector<KmerHit>& rcHits,
                                          int expRChits,
                                          int estInsertSize,
                                          int numBases)
{
    // NB: rcHits must be sorted by location!

    std::vector<Candidate> candidateHits;
    const int nRcHits = boost::numeric_cast<int>(rcHits.size());

    // We may have an estimate of the expected number of RC hits, but we want
    // to look at lesser candidates as well. The only real savings here is
    // analysis time (fewer candidates to consider)
    int densityThreshold = std::min(nRcHits,
                                    std::max(expRChits/4, 6));
    int flankSize = 40;
    auto wBegin = rcHits.begin();
    auto wEnd = rcHits.begin();
    int lastHit = 0;
    for (int i = 0; i < nRcHits; ++i)
    {
        // We don't want to add a bunch of duplicate densities to process
        // later:
        if (lastHit == rcHits[i].Location())
            continue;
        lastHit = rcHits[i].Location();
        while (wBegin->Location() + flankSize < rcHits[i].Location())
            ++wBegin;
        while (rcHits[i].Location() + flankSize >= wEnd->Location()
                && wEnd != rcHits.end())
            ++wEnd;
        assert(wEnd > wBegin);
        int density = wEnd - wBegin;
        int scaledThreshold = densityThreshold;
        if (rcHits[i].Location() < estInsertSize)
        {
            scaledThreshold *= rcHits[i].Location();
            scaledThreshold /= estInsertSize;
        }
        else if (numBases - rcHits[i].Location() < estInsertSize)
        {
            scaledThreshold *= numBases - rcHits[i].Location();
            scaledThreshold /= estInsertSize;
        }

        if (density >= scaledThreshold)
        {
            candidateHits.emplace_back(density, rcHits[i].Location());
        }
    }
    return candidateHits;
}

int GetOr(const std::vector<int>& map,
          int key, int defaultValue=0)
{
    if (static_cast<size_t>(key) < map.size())
        return map[key];
    return defaultValue;
}

int EstMaxLookback(int numBases, int estInsertSize)
{
    return std::max(numBases / 2, estInsertSize * 2);
}

std::vector<int> FindBestAdapters(
        const std::vector<Candidate>& candidateHits,
        const std::vector<int>& fwLengthWeights,
        int fwLengthWeightNorm,
        int numBases,
        int estInsertSize)
{
    // find the min distance/max score set of candidates using the same algo
    // that AdapterFinder uses to select candidates:
    //
    // NB: Obviously this is O(n^2). However, the list of densities is quite
    // small as the number of kmer hits is small. The number of hits with
    // sufficient surrounding density is even smaller, and duplicate hits are
    // thrown out. The list is usually only a few dozen densities.

    const int nCandidateHits = boost::numeric_cast<int>(candidateHits.size());
    std::vector<int> scores;
    std::vector<int> moves;

    int maxLookback = EstMaxLookback(numBases, estInsertSize);

    int canonicalScore = GetOr(fwLengthWeights, estInsertSize);
    // Add a possible move from the end, to prevent false positives getting the
    // highest score in the final stretch just from not being too far from the
    // penultimate adapter:
    for (int i = 0; i <= nCandidateHits; ++i)
    {
        int densityLocation = numBases;
        int density = 0;
        bool isEnd = false;
        if (i < nCandidateHits)
        {
            densityLocation = candidateHits[i].location;
            density = candidateHits[i].density;
        }
        else
        {
            isEnd = true;
        }
        // fwLengthWeights amplitudes are determined by polymerase read length,
        // not insert length. Hit count is determined by insert length. We want
        // these to be on the same scale
        int max = GetOr(fwLengthWeights, densityLocation) / fwLengthWeightNorm;

        // if we're less than an insert from the beginning, we don't want to
        // support an intervening false positive
        if (densityLocation < estInsertSize)
        {
            max = (canonicalScore * densityLocation)
                  / (fwLengthWeightNorm * estInsertSize);
        }

        int move = -1;
        for (int j = 0; j < i; ++j)
        {
            int prevLocation = candidateHits[j].location;
            int dist = densityLocation - prevLocation;
            if (dist > maxLookback)
               continue;
            else if (dist < estInsertSize / 2.2 && !isEnd)
                break;
            int prevScore = scores[j];

            int distBonus = GetOr(fwLengthWeights, dist);

            // If our estimate is a little long, we might overshoot and skip
            // the last adapter if it is really close to the end.
            if (isEnd)
            {
                double insertFraction = (
                        1.0 - std::min(1.0, diff(dist, estInsertSize)
                           / static_cast<double>(estInsertSize)));
                distBonus = std::lround(
                        canonicalScore * insertFraction);
                // We want to penalize overly long estimates, however:
                if (dist > estInsertSize)
                    distBonus /= 10;
            }

            int score = prevScore + distBonus / fwLengthWeightNorm;
            if (score > max)
            {
                max = score;
                move = j;
            }
        }
        // If we're too close to the end, we don't want a bunch of local
        // density adding a false positive
        max += density;
        scores.push_back(max);
        moves.push_back(move);
    }

    int start = -1;
    int max = std::numeric_limits<int>::min();
    for (size_t i = 0; i < scores.size(); ++i)
    {
        if (scores[i] > max)
        {
            max = scores[i];
            start = i;
        }
    }

    std::vector<int> candidates;
    // We no longer include non-candidate hits in densities, so even the first
    // entry can be a candidate. This is especially true if the first candidate
    // is the max!
    int mi = start;
    while (mi >= 0)
    {
        // We included the last base to prevent false positives in the last
        // stretch, but we don't really want an adapter candidate there...
        if (mi < nCandidateHits)
        {
            candidates.push_back(candidateHits[mi].location);
        }
        mi = moves[mi];
    }
    std::reverse(candidates.begin(), candidates.end());
    return candidates;
}

// We re-sort rcHits mulitple times, therefore no const
std::vector<int> RefineAdapters(const std::vector<int>& candidates,
                                std::vector<KmerHit>& rcHits,
                                int estInsertSize,
                                int kmerSize,
                                int centerSampleSize,
                                int expectedPalindromeDiffusionRadius)
{
    std::vector<std::tuple<int, int>> centers;

    for (size_t ci = 0; ci < candidates.size(); ++ci)
    {
        int mode = candidates[ci];

        // We now want to refine the estimate above by taking those hits
        // closest to our estimate and finding the median position that they
        // suggest. To do this we need to sort rcHits:
        auto distFrom = [](const int val,
                           const KmerHit& hit)
        {
            int revDist = diff(hit.reverse, val);
            int forDist = diff(hit.forward, val);
            return revDist + forDist;
        };

        auto firstCloser = [distFrom, kmerSize](const int val,
                           const KmerHit& first,
                           const KmerHit& second,
                           bool rejectAdjacent) -> bool
        {
            // If either is an anomalous or synthetic hit, exclude it
            if (rejectAdjacent)
            {
                if (first.reverse - first.forward <= kmerSize)
                    return false;
                if (second.reverse - second.forward <= kmerSize)
                    return true;
            }
            return (distFrom(val, first) < distFrom(val, second));
        };

        auto distFromMode = [firstCloser, mode](const KmerHit& first,
                                                const KmerHit& second) -> bool
        { return firstCloser(mode, first, second, false); };

        int numsamples = std::min(
                boost::numeric_cast<int>(rcHits.size()),
                centerSampleSize);
        // TODO (MDS 20200602): Order matters if two choices receive the same
        // amount of support. For some reason a unit test breaks. Figure out a
        // better way to pick between the choices, figure out what is wrong with
        // that silly test. Make this a set. It will only ever contain three
        // values from the above analysis, but two are almost always the same,
        // wasting time
        std::vector<int> choices;

        // we rely on everything less than the nth_element being less than the
        // n_th element, but we don't care about specific order, so we don't
        // have to do a full sort:
        std::nth_element(rcHits.begin(), rcHits.begin() + numsamples,
                         rcHits.end(), distFromMode);
        // We do want to more aggresively sample the closest samples, so we'll
        // sort the closest numsamples hits:
        std::sort(rcHits.begin(), rcHits.begin() + numsamples, distFromMode);
        std::vector<double> closeHits;
        for (int i = 0; i < numsamples; ++i)
        {
            double hitLocation = static_cast<double>(
                ((rcHits[i].reverse + rcHits[i].forward)/2));
            if (centers.size() > 0)
            {
                int lastCenter = std::get<0>(centers.back());
                if (hitLocation < lastCenter + estInsertSize/3)
                {
                    continue;
                }
            }
            if (hitLocation > mode + estInsertSize/3
                    || (mode > estInsertSize/3
                        && hitLocation < mode - estInsertSize/3))
            {
                continue;
            }
            // If the span of the hit is greater than the expected palindrome
            // size
            if (distFrom(mode, rcHits[i]) > estInsertSize * 2.2)
            {
                continue;
            }
            closeHits.push_back(hitLocation);

            // We sometimes get a burst or noise near the inflection point,
            // meaning we only have a small toe-hold on the real center, and a
            // whole lot of signal for a center that is slightly off. I think
            // it makes sense to also look at the median of the closes hits, to
            // see if there is something there:
            if (closeHits.size() == 10)
            {
                choices.push_back(modifyingQuantile(closeHits, 0.5));
            }
        }
        choices.push_back(mode);

        if (closeHits.size() > 0)
        {
            // Have to sort again for that median, but not too many samples:
            choices.push_back(modifyingQuantile(closeHits, 0.5));
        }

        int support = 0;
        int center = choices[0];
        for (const auto& option : choices)
        {
            // We're going to measure support for this location as the number
            // of close hits that span our estimate. This is of course
            // imperfect, but it works well enough for now.
            auto distFromCenter = [firstCloser, option](const KmerHit& first,
                                           const KmerHit& second) -> bool
            { return firstCloser(option, first, second, true); };

            std::nth_element(rcHits.begin(), rcHits.begin() + numsamples,
                             rcHits.end(), distFromCenter);
            int thisSupport = 0;
            for (int i = 0; i < numsamples; ++i)
            {
                if (rcHits[i].forward < option && rcHits[i].reverse > option)
                    ++thisSupport;
            }
            if (thisSupport > support)
            {
                support = thisSupport;
                center = option;
            }
        }

        // We don't really want to threshold density separately, as that
        // requires tuning two thresholds that obliquely measure the same
        // thing: how many hits support this location. If either is low this
        // should pull us down, but if one is a little low but it really does
        // seem like there is a palindrome, we should be more robust to
        // hit-smearing.

        // The above median finding can move things around a bit. We don't want
        // to add a bunch of duplicates right next to each other, we want to
        // go with the most supported option if possible
        if (centers.size() > 0 && diff(center, std::get<0>(centers.back()))
                <= expectedPalindromeDiffusionRadius)
        {
            // if we're too close and better supported replace
            if (std::get<1>(centers.back()) < support)
                centers.back() = std::make_tuple(center, support);;
            // otherwise ignore this candiate
        }
        else if (support > 1)
        {
            // if there aren't previous accepted candidates or we weren't too
            // close, go ahead and accept the candidate
            //
            // We're more accepting of adapter calls very close to the first or
            // last adapter, as frequently these aren't flanked by enough bases
            // to drum up support. We may want to limit this benefit to
            // candidates at most some distance from the start and end.
            //centers.push_back(std::make_tuple(center, support, density));
            //centers.push_back(center);
            centers.push_back(std::make_tuple(center, support));
        }
    }
    std::vector<int> ret;
    for (const auto& center : centers)
    {
        ret.push_back(std::get<0>(center));
    }
    return ret;
}

std::vector<int> EstimateTransitionScores(
        const std::vector<KmerHit>& rcHits,
        const std::vector<KmerHit>& forwardHits,
        int fwLengthWeightNorm,
        double expFwHits,
        int estInsertSize,
        int maxLookback,
        bool fwHitMethodRejected)
{
    assert(maxLookback > 0);
    assert(fwLengthWeightNorm > 0);
    assert(estInsertSize > 0);
    std::vector<int> fwLengthWeights;
    fwLengthWeights.resize(maxLookback, 0);

    // We want to increase the weight of positions around estimates of insert
    // size, but we don't want to iterate over all of those positions every time
    // we process an estimate. Therefore we accumulate a list of starts (stored
    // in fwLengthWeights) and ends (stored in decrements) of this local
    // increment, and use a final loop to add up the weight at each position
    // (cumulative sum of starts and stops at each position).
    int totAdded = 0;
    if (forwardHits.size() > rcHits.size()/4
            && forwardHits.size() > expFwHits)
    {
        std::vector<int> decrements;
        decrements.resize(maxLookback, 0);
        for (size_t i = 0; i < forwardHits.size(); ++i)
        {
            int localEst = forwardHits[i].Span()/2;
            int localFlank = localEst/10;
            ++fwLengthWeights[std::max(0, localEst - localFlank/2)];
            ++decrements[std::min(maxLookback, localEst + localFlank/2) - 1];
        }
        int curWeight = 0;
        for (size_t i = 0; i < fwLengthWeights.size(); ++i)
        {
            curWeight += fwLengthWeights[i];
            fwLengthWeights[i] = curWeight;
            totAdded += curWeight;
            curWeight -= decrements[i];
            assert(curWeight >= 0);
        }
    }

    // We didn't have enough forward hits to populate the transition score
    // datastructure, or we don't believe the forward hits are consistent with
    // the insert size. We'll impute some forward hit lengths from the adapter
    // hits if necessary:
    if (totAdded == 0 || fwHitMethodRejected)
    {
        int localFlank = estInsertSize/10;
        for (int winI = std::max(0, estInsertSize - localFlank/2);
                winI < estInsertSize + localFlank/2; ++winI)
        {
            if (winI < maxLookback)
                fwLengthWeights[winI] += 100 * fwLengthWeightNorm;
        }
    }
    return fwLengthWeights;
}

void PopulateHits(std::vector<KmerHit>& rcHits,
                  std::vector<KmerHit>& forwardHits,
                  int kmerSize,
                  const std::string& bases)
{
    // We're going to hash a lot of analog kmers into unsigned integers.
    // These integers are equivalent to the index of that
    // kmer's leaf in an array of leaves of a perfect suffix tree of all
    // possible kmers.
    //
    // We could use these indices and a fixed size array of 4^kmerSize, but
    // just resetting the values in said array is more expensive when our
    // average insert size is much less than 4^15, which is a good kmer size.
    //
    // We do this hashing for 3 reasons:
    //  1. Taking a substring, finding its reverse complement and using those
    //  as keys into a hash is expensive.
    //  2. The forward and reverse complement hashes can be calculated for each
    //  base in an insert in linear time with the insert size, and constant
    //  time with the kmer size (by updating the hash, rather than recomputing
    //  it).
    //  3. If we want to move back to an array instead of a map, we can do that
    //  easily.
    //
    // That is a long way of saying we'll need to use unsigned int powers of 4
    // up to kmerSize a lot, so lets just make a lookup table.
    const int numAnalogs = 4;
    const int nBases = boost::numeric_cast<int>(bases.size());
    std::vector<int> powlut(kmerSize + 1, 1l);
    for (int offset = 1; offset <= kmerSize; ++offset)
        powlut[offset] = powlut[offset - 1] * numAnalogs;

    // Obviously a map is not a trie. See the comment above for why the map is
    // a cheaper datastructure for giving us the same effect, depending on
    // insert size and kmer size.
    std::unordered_map<int, int> maptrie;
    maptrie.reserve(nBases);

    // Non-unique kmers are generally worse (see the BELLA paper, Guidi et al.
    // 2018). This would be a bit more expensive, however, and would really
    // screw up if you have multiple passes. It also doesn't make sense to
    // score multiple hits for repeated kmers if there are multiple passes
    // present, as their suggested midpoints would be bogus.
    int fdIndex, rcIndex;
    std::tie(fdIndex, rcIndex) = HashKmer(bases, 0, kmerSize, powlut);
    for (int i = 0; i < nBases - kmerSize; ++i)
    {
        if (i > 0)
        {
            std::tie(fdIndex, rcIndex) = UpdateKmerHash(
                bases, i, kmerSize, powlut, fdIndex, rcIndex);
        }

        // Store the forward Kmer
        auto fwHit = maptrie.find(fdIndex);
        // We don't really want to deal in anything less than 50 base inserts,
        // a forward hit spanning a palindrome should be at least 100 bases
        if (fwHit != maptrie.end() && i - fwHit->second > 100)
        {
            forwardHits.emplace_back(i, fwHit->second);
        }
        maptrie[fdIndex] = i;

        // See if the RC Kmer has a match to some previous forward Kmer, as
        // long as they don't overlap
        auto rcHit = maptrie.find(rcIndex);
        if (rcHit != maptrie.end() && i - rcHit->second >= kmerSize)
        {
            // I'm adding kmerSize here so the location estimate is correct.
            // That is, we're storing the forward and reverse outermost points
            // of the two kmers, relative to the inflection point. Therefore
            // the average of these two points is the suggested
            // inflection point, regardless of the kmer size used to identify
            // the points. We subtract one, however, as we want the reverse
            // index to be inclusive
            rcHits.emplace_back(i + kmerSize - 1, rcHit->second);
            // we only want to use each hit once, or you will get a burst
            // hitting on a single foward homopolymer
            maptrie.erase(rcHit);
        }
    }
}
} // anonymous

//std::vector<std::tuple<int, int, int>>
std::vector<int>
PalindromeTrieCenter(const std::string& bases,
                     const std::vector<RegionLabel>& adapters,
                     int hqrOffset,
                     int kmerSize,
                     int centerSampleSize)
{
    const int nBases = boost::numeric_cast<int>(bases.size());

    // If there is no work to do...
    if (nBases < kmerSize)
        return {};

    // rcHits will hold the actual kmer locations, because we want to weight
    // kmers closer to the adapter location higher than those farther away
    // (which can be biased by insertions or bursts)
    std::vector<KmerHit> rcHits;
    rcHits.reserve(bases.size());
    std::vector<KmerHit> forwardHits;
    forwardHits.reserve(bases.size());
    PopulateHits(rcHits, forwardHits, kmerSize, bases);

    // More than one palindrome can exist in a sequence, but the adapter point
    // should have the most hits. We will estimate the location using kernel
    // smoothing and the mode, and refine it later.

    // If our hypothesis is that we're observing so many passes, we would
    // expect a number of hits for each pass in a certain area. The number
    // depends strongly on accuracy, but we can estimate accuracy using the
    // average distance between kmer hits.
    double expectedAccuracy = EstimateAccuracy(rcHits, forwardHits);
    double expFwHits = pow(expectedAccuracy, 2.0 * kmerSize)
                        * nBases / 3;

    InsertSizeEstimate insertEstDetails = EstimateInsertSize(
            rcHits, forwardHits, adapters, bases, expFwHits, hqrOffset);
    int estInsertSize = insertEstDetails.size;

    if (!insertEstDetails.successful)
    {
        std::vector<int> ret;
        for (const auto& adp : adapters)
            ret.push_back(adp.begin);
        return ret;
    }

    // Espected reverse compliment hit density depends on insert size, while
    // forward length frequency depends on the number of bases and the number
    // of passes. We don't want one score to dominate the other, so we
    // normalize the transition scores.
    //
    // This is essentially the expected number of forward hits of the correct
    // length:
    int fwLengthWeightNorm = std::lround((nBases - estInsertSize)
                             * pow(expectedAccuracy, 2.0 * kmerSize));
    // But we don't want the term normalized to near one, our densities tend to
    // be ~100, so we'll dial in a "transition score weight" term:
    fwLengthWeightNorm /=  60;

    fwLengthWeightNorm = std::max(1, fwLengthWeightNorm);

    // Next we want to setup a datastructure to store transition scores of
    // sorts, to weight accepting adapter locations a certain distance apart
    int maxLookback = EstMaxLookback(bases.size(), estInsertSize);
    const auto fwLengthWeights = EstimateTransitionScores(
        rcHits, forwardHits, fwLengthWeightNorm, expFwHits, estInsertSize,
        maxLookback, insertEstDetails.fwHitMethodRejected);

    int expectedPalindromeDiffusionRadius = estInsertSize / 8;

    // Rough estimate of the number of times you will correctly sequence each
    // kmer twice. We don't account for the kernel smoothing/flankSize, because
    // we expect only this number of total RC hits for the palindrome. The
    // location to which they point can be affected by bursts etc. that
    // wouldn't impact their existance.
    int expRChits = std::lround(pow(expectedAccuracy, 2.0 * kmerSize)
                               * estInsertSize);

    // We usually really like existing adapters. We make sure they are well
    // represented in the reverse complement hit density scores by adding some
    // fake hits in the right places
    // We'll ramp up the minimum score of this bias with total adapter counts
    int priorMaxWeight = 100;
    int priorMinWeight = 26;
    if (adapters.size() == 1)
        priorMinWeight = 0;
    if (adapters.size() == 2)
        priorMinWeight = 13;
    for (const auto& adp : adapters)
    {
        int nhits = (adp.accuracy > 0.75) ? expRChits * adp.accuracy : 0;
        nhits += expRChits * adp.flankingScore/300;
        nhits = std::min(priorMaxWeight, std::max(priorMinWeight, nhits));

        // Only synthesize support for adapters found via alignment (and
        // halfway decent ones, because why not)
        if (adp.accuracy > 0.60)
        {
            int adpLocation = (adp.end + adp.begin)/2 - hqrOffset;
            int radius = std::min({adpLocation,
                                   expectedPalindromeDiffusionRadius,
                                   nBases - adpLocation});
            for (int i = 0; i < nhits; ++i)
            {
                rcHits.emplace_back(adpLocation + radius,
                                     adpLocation - radius);
            }
        }
    }

    // If we don't find any hits!
    if (rcHits.size() == 0)
        return {};

    auto spanSorter = [](const KmerHit& first,
                         const KmerHit& second) -> bool
    { return first.Span() < second.Span(); };

    // We want to eliminate rc hits over a certain size, because they certainly
    // happen and can be very misleading:
    std::sort(rcHits.begin(), rcHits.end(), spanSorter);
    auto tooBig = rcHits.begin();
    while (tooBig->reverse - tooBig->forward < 3 * estInsertSize
            && tooBig != rcHits.end())
        ++tooBig;
    if (tooBig != rcHits.begin() && tooBig != rcHits.end())
        rcHits.erase(tooBig, rcHits.end());

    // Lets find the density center. We used to take the straight mode, but
    // that could be misled by local palindromes, especially when the real
    // center has some diffusesness. Instead we generate a smoothed reverse
    // complement hit density profile, and pick the best chain of locations
    auto locationSorter = [](const KmerHit& first,
                             const KmerHit& second) -> bool
    { return first.Location() < second.Location(); };

    std::sort(rcHits.begin(), rcHits.end(), locationSorter);

    std::vector<Candidate> candidateHits = SmoothHitDensities(
        rcHits, expRChits, estInsertSize, bases.size());

    std::vector<int> candidates = FindBestAdapters(
        candidateHits, fwLengthWeights, fwLengthWeightNorm, bases.size(),
        estInsertSize);

    auto tbr = RefineAdapters(candidates, rcHits, estInsertSize, kmerSize,
            centerSampleSize, expectedPalindromeDiffusionRadius);
    // Add the HQRegion start offset to each center, so they're in the same
    // coordinate system as the existing adapters
    for (auto& val : tbr)
    {
        val += hqrOffset;
    }
    return tbr;
}

}}} // ::PacBio::Primary::Postprimary
