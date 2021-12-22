// Copyright (c) 2014-2020, Pacific Biosciences of California, Inc.
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

#pragma once

#include <assert.h>
#include <iostream>
#include <limits>
#include <memory>

#include <spoa/spoa.hpp>

#include <pacbio/logging/Logger.h>
#include <bazio/FastaEntry.h>
#include <bazio/RegionLabel.h>
#include <bazio/RegionLabelType.h>

#include <postprimary/alignment/PairwiseAligner.h>
#include <postprimary/alignment/ScoringScheme.h>
#include <postprimary/alignment/ScoringScope.h>
#include <postprimary/bam/Platform.h>

#include "SpacedSelector.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

namespace {

std::string GetPreFlank(const std::string& bases,
                        const Cell& candidate,
                        int flankLength,
                        bool compressHomopolymers,
                        int inset)
{
    int canStart = static_cast<int>(candidate.jBeginPosition);
    if (!compressHomopolymers)
    {
        int start = (canStart >= flankLength)
            ? canStart - flankLength : 0;
        start += inset;
        int length = (canStart >= flankLength)
            ? flankLength : canStart;
        return bases.substr(start, length);
    }
    std::string ret = "";
    int seen = 0;
    int start = (canStart > 0) ? canStart - 1 : 0;
    start += inset;
    size_t index = start;
    while (seen < flankLength)
    {
        if (ret.empty() || bases[index] != ret.back())
        {
            ret += bases[index];
            seen++;
        }
        if (index == 0)
            break;
        index--;
    }
    std::reverse(ret.begin(), ret.end());
    return ret;
};

std::string GetPostFlank(const std::string& bases,
                         const Cell& candidate,
                         int flankLength,
                         bool compressHomopolymers,
                         int inset)
{
    int start = static_cast<int>(candidate.jEndPosition);
    start = (start >= inset) ? start - inset : 0;
    if (!compressHomopolymers)
    {
        auto length = std::min(static_cast<int>(bases.size() - candidate.jEndPosition), flankLength);
        return  bases.substr(start, length);
    }
    std::string ret = "";
    int seen = 0;
    size_t index = start;
    while (seen < flankLength && index < bases.size())
    {
        if (ret.empty() || bases[index] != ret.back())
        {
            ret += bases[index];
            seen++;
        }
        index++;
    }
    return ret;
};

}

using namespace PacBio::Primary;

/// Labels a single polymerase read with adapter RegionLabels
class AdapterLabeler
{
public: // structors
    AdapterLabeler(std::shared_ptr<std::vector<FastaEntry>>& adapterList
                   , Platform platform
                   , float minSoftAccuracy = -1.0
                   , float minHardAccuracy = -1.0
                   , float minFlankingScore = -1.0
                   , bool localAlnFlanking = false
                   , int flankLength = -1
#ifdef DIAGNOSTICS
                   , bool emitAdapterMetrics = false
#endif
                   , bool lookForLoop = false
                   , bool trimToLoop = false
                   , int stemLength = 0
                   )
        : adapterList_(adapterList)
        , localAlnFlanking_(localAlnFlanking)
#ifdef DIAGNOSTICS
        , emitAdapterMetrics_(emitAdapterMetrics)
#endif
        , lookForLoop_(lookForLoop)
        , trimToLoop_(trimToLoop)
        , stemLength_(stemLength)
    {
        Init(platform);
        if (minSoftAccuracy > 0.0)
            minSoftAccuracy_ = minSoftAccuracy;
        if (minHardAccuracy > 0.0)
            minHardAccuracy_ = minHardAccuracy;
        if (minFlankingScore > 0.0)
            minFlankingScore_ = minFlankingScore;
        if (flankLength > 0)
            flankLength_ = flankLength;
        if (trimToLoop)
            PBLOG_INFO << "An adapter caller has been configured "
                          "with the trimToLoop method";
        if (lookForLoop)
            PBLOG_INFO << "An adapter caller has been configured "
                          "with the lookForLoop method";
        if (!trimToLoop && !lookForLoop)
            PBLOG_INFO << "An adapter caller has been configured "
                          "with the standard method";
    }

    /// Only for debug purposes
    AdapterLabeler(std::shared_ptr<std::vector<FastaEntry>>& adapterList,
                   int minAdapterScore, int minAdapterSpacing,
                   float minFlankingScore, float minSoftAccuracy,
                   float minHardAccuracy, ScoringScheme adapterFindingScoring,
                   ScoringScheme flankingRegionsScoring)
        : adapterList_(adapterList)
        , minAdapterScore_(minAdapterScore)
        , minAdapterSpacing_(minAdapterSpacing)
        , minFlankingScore_(minFlankingScore)
        , minSoftAccuracy_(minSoftAccuracy)
        , minHardAccuracy_(minHardAccuracy)
        , adapterFindingScoring_(adapterFindingScoring)
        , flankingRegionsScoring_(flankingRegionsScoring)
        , flankLength_(70)
        , localAlnFlanking_(false)
#ifdef DIAGNOSTICS
        , emitAdapterMetrics_(true)
#endif
        , lookForLoop_(false)
        , trimToLoop_(false)
        , stemLength_(0)
    {}

    // Move constructor
    AdapterLabeler(AdapterLabeler&&) = delete;
    // Copy constructor
    AdapterLabeler(const AdapterLabeler&) = delete;
    // Move assignment operator
    AdapterLabeler& operator=(AdapterLabeler&&) = delete;
    // Copy assignment operator
    AdapterLabeler& operator=(const AdapterLabeler&) = delete;
    // Destructor
    ~AdapterLabeler() = default;

public: // non-modifying methods
    /// Labels a single polymerase read with adapter RegionLabels
    std::vector<RegionLabel> Label(const std::string& bases,
                                   RegionLabel& hqRegion,
                                   bool extend = true,
                                   int stemLengthOverride = -1) const
    {
        std::vector<RegionLabel> output;

        // Symmetric adapters
        if (adapterList_->size() == 1)
        {
            const auto& extHQRegion = extend ?
                ExtendHQRegionWithAdapter(bases, hqRegion, 0) : hqRegion;
            output = LabelSingleAdapter(bases, 0, extHQRegion, minAdapterSpacing_, stemLengthOverride);

            // Cut overlapping adapters
            if (output.size() > 1)
            {
                for (size_t i = 1; i < output.size(); ++i)
                    RegionLabel::CutRegionLabels(&output.at(i - 1), &output.at(i));

                // Sort adapters by begin
                std::stable_sort(output.begin(), output.end(), RegionBeginComparer);

                // Repeat the cut to handle regions that were formerly-contained regions
                for (size_t i = 1; i < output.size(); ++i)
                    RegionLabel::CutRegionLabels(&output.at(i - 1), &output.at(i));

            }
            if (output.size() > 0)
            {
                UpdateHQRegionWithAdapterHits(hqRegion, output);
            }
        }
            // Asymmetric adapters
        else
        {
            // Increase spacing as adapters are asymmetric
            int perAdapterSpacing = 2 * minAdapterSpacing_;
            // Get labels from first adapter
            const auto& extHQRegionFirst = extend ?
                ExtendHQRegionWithAdapter(bases, hqRegion, 0) : hqRegion;
            output = LabelSingleAdapter(bases, 0, extHQRegionFirst,
                                        perAdapterSpacing, stemLengthOverride);
            // Get labels from second adapter
            const auto& extHQRegionSecond = extend ?
                ExtendHQRegionWithAdapter(bases, hqRegion, 1) : hqRegion;
            auto secondAdapterHits = LabelSingleAdapter(
                    bases, 1, extHQRegionSecond, perAdapterSpacing, stemLengthOverride);

            SimplifyOverlappingIntervals(&output, &secondAdapterHits);

            // Append secondAdapterHits to output via move semantyics
            output.reserve(output.size() + secondAdapterHits.size());
            std::move(secondAdapterHits.begin(), secondAdapterHits.end(),
                      std::back_inserter(output));
            secondAdapterHits.clear();

            // Sort adapters by begin
            std::stable_sort(output.begin(), output.end(), RegionBeginComparer);

            // Cut overlapping adapters
            if (output.size() > 1)
            {
                for (size_t i = 1; i < output.size(); ++i)
                    RegionLabel::CutRegionLabels(&output.at(i - 1), &output.at(i));

                // Sort adapters by begin
                std::stable_sort(output.begin(), output.end(), RegionBeginComparer);

                // Repeat the cut to handle regions that were formerly-contained regions
                for (size_t i = 1; i < output.size(); ++i)
                    RegionLabel::CutRegionLabels(&output.at(i - 1), &output.at(i));

            }
            if (output.size() > 0)
            {
                UpdateHQRegionWithAdapterHits(hqRegion, output);
            }
        }
        return output; // Implicit move semantics take care that RegionLabels are moved.
    }


    static void SimplifyOverlappingIntervals(std::vector<RegionLabel>* intervals1,
                                             std::vector<RegionLabel>* intervals2)
    {
        // Where two sets of adapter labels conflict, we take the highest accuracy one
        for (auto it1 = intervals1->begin(), it2 = intervals2->begin(); 
             it1 != intervals1->end() && it2 != intervals2->end();) {
            // If the two labels don't overlap, advance the left-most one
            if (it1->end < it2->begin) {
                ++it1;
                continue;
            } else if (it2->end < it1->begin) {
                ++it2;
                continue;
            }

            // If they do overlap, delete the lowest scoring label
            if (it1->begin <= it2->end && it2->begin <= it1->end) {
                if (it1->score >= it2->score) {
                    it2 = intervals2->erase(it2);
                } else {
                    it1 = intervals1->erase(it1);
                }
            }
        }
    }

private: // data
    const std::shared_ptr<std::vector<FastaEntry>> adapterList_;
    int minAdapterScore_;
    int minAdapterSpacing_;
    float minFlankingScore_;
    float minSoftAccuracy_;
    float minHardAccuracy_;
    ScoringScheme adapterFindingScoring_;
    ScoringScheme flankingRegionsScoring_;
    uint32_t flankLength_;
    bool localAlnFlanking_;
#ifdef DIAGNOSTICS
    bool emitAdapterMetrics_;
#endif
    bool lookForLoop_;
    bool trimToLoop_;
    int stemLength_;


private:
    void Init(const Platform platform)
    {
        switch (platform)
        {
            case Platform::RSII:
                minAdapterScore_ = 0;
                minAdapterSpacing_ = 40;
                minFlankingScore_ = 10;
                minSoftAccuracy_ = 0.66;
                minHardAccuracy_ = 0.60;
                flankLength_ = 70;
                break;
            case Platform::SEQUELII:
                // we don't have sequelII specific params yet, fall through
            case Platform::SEQUEL:
                minAdapterScore_ = 0;
                minAdapterSpacing_ = 40;
                minFlankingScore_ = 110;
                minSoftAccuracy_ = 0.72;
                minHardAccuracy_ = 0.55;
                flankLength_ = 70;
                break;
            case Platform::MINIMAL:
                minAdapterScore_ = 0;
                minAdapterSpacing_ = 40;
                minFlankingScore_ = 10;
                minSoftAccuracy_ = 0.40;
                minHardAccuracy_ = 0.40;
                flankLength_ = 70;
                break;
            case Platform::NONE:
            default:
                throw std::runtime_error("Platform not specified!");
                break;
        }
        adapterFindingScoring_ = ScoringScheme(platform, ScoringScope::ADAPTER_FINDING);
        flankingRegionsScoring_ = ScoringScheme(platform, ScoringScope::FLANKING_REGIONS);
    }

    Cell AlignSuspectedStem(const std::string& adapterStem, const std::string& candidateStem) const
    {
        std::unique_ptr<int32_t[]> matrix{PairwiseAligner::SWComputeMatrix(
            adapterStem, candidateStem, adapterFindingScoring_)};

        // DP constants
        const int32_t M = adapterStem.size() + 1;
        const int32_t N = candidateStem.size() + 1;
        const int32_t beginLastRow = (M - 1) * N;
        int maxJ = 0;
        for (int j = 0; j < N; ++j)
        {
            if (matrix[beginLastRow + j] > matrix[beginLastRow + maxJ])
                maxJ = j;
        }
        Cell ret(maxJ, matrix[maxJ]);
        ret.score = matrix[beginLastRow + maxJ];
        PairwiseAligner::SWBacktracking(matrix.get(), ret, N, adapterStem, candidateStem,
                                        adapterFindingScoring_);
        return ret;
    };

    float ScaledMinSoftAccuracy(int adapterLength) const
    {
        // The adapter hit filters were tuned for a 45bp adapter
        const int lengthDelta = 45 - adapterLength;
        return minSoftAccuracy_ + 0.01f * lengthDelta;
    }

    int ScaledMinFlankingScore(int adapterLength) const
    {
        // The adapter hit filters were tuned for a 45bp adapter
        const int lengthDelta = 45 - adapterLength;
        return minFlankingScore_ + lengthDelta;
    }

    void UpdateHQRegionWithAdapterHits(RegionLabel& hqr, const std::vector<RegionLabel>& adapterHits) const
    {
        const auto& firstAdapterHit = adapterHits.front();
        const auto& lastAdapterHit = adapterHits.back();

        if (firstAdapterHit.begin < hqr.begin)
        {
            hqr.begin = firstAdapterHit.begin;
        }

        if (hqr.end < lastAdapterHit.end)
        {
            hqr.end = lastAdapterHit.end;
        }
    }

    RegionLabel ExtendHQRegionWithAdapter(const std::string& bases, const RegionLabel& hqr, const size_t adapterNumber) const
    {
        assert(hqr.type == RegionLabelType::HQREGION);

        // adapter string&
        const auto& adapter = adapterList_->at(adapterNumber).sequence;
        // adapter length
        const auto adapterLength = adapter.size();
        const float adapterLengthFudgeFactor = 1.5f;
        const int adapterLengthFudge = std::ceil(adapterLength * adapterLengthFudgeFactor);

        // Extend left boundary of HQ-region.
        const int hqStart = std::max(hqr.begin - adapterLengthFudge, 0);

        // Extend right boundary of HQ-region.
        const int hqEnd = std::min(hqr.end + adapterLengthFudge, static_cast<int>(bases.size()));

        RegionLabel extHQRegion(hqStart, hqEnd, hqr.score, hqr.sequenceId, hqr.type);

        return extHQRegion;
    }

    /// Labels a single polymerase read with adapte RegionLabels
    /// given a single adapter and extended hqRegion.
    //
    // If we've determined that there is or is not a stem sequence, we want to
    // lock the adapter sequence.
    //
    // 1. If we haven't locked but we're not initially
    // configured to look for loop, we're always looking for the supplied
    // adapter sequence.
    //
    // 2. If we haven't locked but we are looking for loop, we look for the
    // adapter sequence +/- the stem length on each side
    //
    // 3. If we have locked on the full sequence, then we know the stem length
    // is 0. lookForLoop and trimToLoop should be set to off, and the stem
    // length override should be used
    // (SensitiveAdapterSearch)
    //
    // 4. If we have locked on the loop-only sequence, then we know the stem
    // length is X, lookForLoop and trimToLoop should be set to off, default
    // stemLength should be used (no stem length override)
    //
    // 5. Scoring hyperparameters will scale with actual searched stem length
    // automatically

    std::vector<RegionLabel> LabelSingleAdapter(const std::string& bases,
                                                const size_t adapterNumber,
                                                const RegionLabel& extHQRegion,
                                                const int minAdapterSpacing,
                                                const int stemLengthOverride = -1) const
    {

        if (lookForLoop_)
            return LabelSingleAdapterLookForLoop(
                    bases, adapterNumber, extHQRegion, minAdapterSpacing,
                    stemLengthOverride);
        if (trimToLoop_)
            return LabelSingleAdapterTrimToLoop(
                    bases, adapterNumber, extHQRegion, minAdapterSpacing,
                    stemLengthOverride);

        std::vector<RegionLabel> output;

        // There are three options for stemLengthOverride:
        // -1: the default. We're not DAFTing, so we use the
        //     default stem length configured in the class (from the CLI).
        //     This will be 8 if DAFT was specified (but we're
        //     in this mithod because sensitiveAdapterCorrection doesn't use
        //     DAFT), or 0 if DAFT was not specified
        //  0: We're in sensitive adapter search for a ZMW with stems
        // ~8: We're in sensitive adapter search for a ZMW without stems
        const int stemLength = (stemLengthOverride >= 0) ? stemLengthOverride : stemLength_;
        bool hasStem = stemLength == 0 ? true : false;

        const auto& adapter = [&](){
            const auto& fullAdapter = adapterList_->at(adapterNumber).sequence;
            return fullAdapter.substr(stemLength, fullAdapter.size() - 2 * stemLength);
        }();

        // adapter length
        const auto adapterLength = adapter.size();
        // bases length
        // const auto length = bases.size();

        const auto& hqRegionBases = bases.substr(extHQRegion.begin, extHQRegion.end - extHQRegion.begin);

        // Fill matrix
        std::unique_ptr<int32_t[]> matrix{PairwiseAligner::SWComputeMatrix(
            adapter, hqRegionBases, adapterFindingScoring_)};

        // DP constants
        const int32_t M = adapterLength + 1;
        const int32_t N = hqRegionBases.size() + 1;
        const int32_t beginLastRow = (M - 1) * N;

        // Convert last row to pairs of position/score
        std::vector<Cell> lastRow;
        lastRow.reserve(N);
        for (int j = 0; j < N; ++j)
        {
            // Filter by adapter score
            if (matrix[beginLastRow + j] > minAdapterScore_)
                lastRow.emplace_back(j, matrix[beginLastRow + j]);
        }

        std::vector<Cell> acceptedCells;
        // Only continue if there are scores above the threshold
        if (!lastRow.empty())
        {
            // Find possible adapter end positions with minimal spacing
            auto cells = SpacedSelector::FindEndPositions(
                std::move(lastRow), minAdapterSpacing);

            // For each possible adapter end position
            float acc;
            for (auto& cell : cells)
            {
                // Perform backtracking
                PairwiseAligner::SWBacktracking(
                        matrix.get(), cell, N, adapter,
                        hqRegionBases, adapterFindingScoring_);
                // Compute accuracy
                acc = Accuracy(cell);

#ifdef DEBUG_TOOLS
                // In debugMode, calculate and record all flanking scores
                PairwiseAligner::FlankingScore(
                    hqRegionBases, cell, flankingRegionsScoring_,
                    localAlnFlanking_, flankLength_);

                acceptedCells.push_back(cell);
#else
#ifdef DIAGNOSTICS
                // In DIAGNOSTICS mode, calculate all flanking scores if
                // they're being emitted. They will be recalculated if
                // necessary later on, so don't emitAdapterMetrics when
                // timing is critical
                if (emitAdapterMetrics_)
                    PairwiseAligner::FlankingScore(
                        hqRegionBases, cell, flankingRegionsScoring_,
                        localAlnFlanking_, flankLength_);
#endif
                // Otherwise only calculate the flanking score if needed
                // Hard filtering
                if (acc < minHardAccuracy_) continue;


                // Soft filtering
                if (acc >= ScaledMinSoftAccuracy(adapterLength)
                    || (PairwiseAligner::FlankingScore(
                            hqRegionBases, cell, flankingRegionsScoring_,
                            localAlnFlanking_, flankLength_),
                        cell.flankingScore >= ScaledMinFlankingScore(adapterLength)))
                {
                    acceptedCells.push_back(cell);
                }
#endif
            }
        }

        for (const auto& cell : acceptedCells)
        {
            // Create adapter label
            output.emplace_back(
                cell.jBeginPosition + extHQRegion.begin,
                cell.jEndPosition + extHQRegion.begin,
                cell.score, adapterNumber, RegionLabelType::ADAPTER);
            output.back().accuracy = Accuracy(cell);
            output.back().flankingScore = cell.flankingScore;
            output.back().hasStem = hasStem;
        }

        return output;
    }

    // This is not currently used, but isn't an unreasonable fallback...
    std::vector<RegionLabel> LabelSingleAdapterLookForLoop(
            const std::string& bases,
            const size_t adapterNumber,
            const RegionLabel& extHQRegion,
            const int minAdapterSpacing,
            const int stemLengthOverride = -1) const
    {
        std::vector<RegionLabel> output;
        bool hasStem = false;

        const int stemLength = (stemLengthOverride >= 0) ? stemLengthOverride : stemLength_;

        const auto& fullAdapter = adapterList_->at(
                adapterNumber).sequence;
        const auto& adapter = fullAdapter.substr(stemLength,
                                                 fullAdapter.size() - 2 * stemLength);

        const auto& adapterStem = [&fullAdapter = adapterList_->at(adapterNumber).sequence,
                                   &stemLength]()
        {
            return fullAdapter.substr(fullAdapter.size() - stemLength, stemLength);
        }();

        // adapter length
        const auto adapterLength = adapter.size();

        const auto& hqRegionBases = bases.substr(extHQRegion.begin,
                                                 extHQRegion.end - extHQRegion.begin);

        // Fill matrix
        std::unique_ptr<int32_t[]> matrix{PairwiseAligner::SWComputeMatrix(
            adapter, hqRegionBases, adapterFindingScoring_)};

        // DP constants
        const int32_t M = adapterLength + 1;
        const int32_t N = hqRegionBases.size() + 1;
        const int32_t beginLastRow = (M - 1) * N;

        std::vector<Cell> loopCells;

        // Convert last row to pairs of position/score
        std::vector<Cell> lastRow;
        lastRow.reserve(N);
        for (int j = 0; j < N; ++j)
        {
            // Filter by adapter score
            if (matrix[beginLastRow + j] > minAdapterScore_)
                lastRow.emplace_back(j, matrix[beginLastRow + j]);
        }

        float meanAdapterAcc = 0.0;
        std::vector<Cell> acceptedCells;
        // Only continue if there are scores above the threshold
        if (!lastRow.empty())
        {
            // Find possible adapter end positions with minimal spacing
            auto cells = SpacedSelector::FindEndPositions(
                std::move(lastRow), minAdapterSpacing);

            // For each possible adapter end position
            float acc;
            for (auto& cell : cells)
            {
                // Perform backtracking
                PairwiseAligner::SWBacktracking(matrix.get(), cell, N, adapter, hqRegionBases,
                                                adapterFindingScoring_);
                // Compute accuracy
                acc = Accuracy(cell);

#ifdef DEBUG_TOOLS
                // In debugMode, calculate and record all flanking scores
                PairwiseAligner::FlankingScore(
                    hqRegionBases, cell, flankingRegionsScoring_,
                    localAlnFlanking_, flankLength_);

                acceptedCells.push_back(cell);
#else
#ifdef DIAGNOSTICS
                // In DIAGNOSTICS mode, calculate all flanking scores if
                // they're being emitted. They will be recalculated if
                // necessary later on, so don't emitAdapterMetrics when
                // timing is critical
                if (emitAdapterMetrics_)
                    PairwiseAligner::FlankingScore(
                        hqRegionBases, cell, flankingRegionsScoring_,
                        localAlnFlanking_, flankLength_);
#endif
                // Otherwise only calculate the flanking score if needed
                // Hard filtering
                if (acc < minHardAccuracy_) continue;

                // Soft filtering
                if (acc >= ScaledMinSoftAccuracy(adapterLength)
                    || (PairwiseAligner::FlankingScore(
                            hqRegionBases, cell, flankingRegionsScoring_,
                            localAlnFlanking_, flankLength_),
                        cell.flankingScore >= ScaledMinFlankingScore(adapterLength)))
                {
                    acceptedCells.push_back(cell);
                    meanAdapterAcc += acc;
                }
#endif
            }
        }

        // Stem extension if necessary
        // (the stem will have been included in the adapter seq if
        // lookForLoop_ is false)
        if (!acceptedCells.empty())
        {
            // match, mismatch, gap open, gap extend
            auto alignmentEngine = spoa::createAlignmentEngine(
                    spoa::AlignmentType::kSW, 4, -13, -7);
            auto graph = spoa::createGraph();

            // 50% margin for inserts in the stem and missed loop bases
            int searchBases = static_cast<int>(ceil(stemLength * 1.5));
            // the strings in the tuples are the raw candidate flanks
            std::vector<std::tuple<std::string, Cell>> precedingStemAlignments;
            std::vector<std::tuple<std::string, Cell>> postcedingStemAlignments;
            for (const auto& cell : acceptedCells)
            {
                const auto& preStemFlank = SequenceUtilities::ReverseCompl(
                        GetPreFlank(hqRegionBases, cell, searchBases, false, 0));
                const auto& postStemFlank = GetPostFlank(
                        hqRegionBases, cell, searchBases, false, 0);

                auto preAlignment = alignmentEngine->align(
                    preStemFlank, graph);
                graph->add_alignment(preAlignment, preStemFlank);
                auto postAlignment = alignmentEngine->align(
                    postStemFlank, graph);
                graph->add_alignment(postAlignment, postStemFlank);

                precedingStemAlignments.push_back(
                    std::make_tuple(preStemFlank,
                                    AlignSuspectedStem(adapterStem, preStemFlank)));
                postcedingStemAlignments.push_back(
                    std::make_tuple(postStemFlank,
                                    AlignSuspectedStem(adapterStem, postStemFlank)));
            }

            // Iterate over the outputs, classify the ZMW
            float meanStemAcc = 0;
            //float meanStemLen = 0;

            for (size_t i = 0; i < acceptedCells.size(); ++i)
            {
                meanStemAcc += Accuracy(std::get<1>(precedingStemAlignments[i]));
                //meanStemLen += ReadLength(std::get<1>(precedingStemAlignments[i]));
                meanStemAcc += Accuracy(std::get<1>(postcedingStemAlignments[i]));
                //meanStemLen += ReadLength(std::get<1>(postcedingStemAlignments[i]));
            }
            auto numStems = precedingStemAlignments.size() + postcedingStemAlignments.size();
            float consensusAccuracy = 0.0;
            int consensusAlnLength = 0;
            if (numStems)
            {
                auto consensus = graph->generate_consensus();
                auto consensusAlignment = AlignSuspectedStem(adapterStem, consensus);
                consensusAccuracy = Accuracy(consensusAlignment);
                consensusAlnLength = ReadLength(consensusAlignment);
            }
            if (numStems)
            {
                meanStemAcc /= numStems;
                //meanStemLen /= numStems;
            }
            if (acceptedCells.size())
                meanAdapterAcc /= acceptedCells.size();
            // A simple, but somewhat crude decision tree
            float consensusAccuracyThresh = 0.80;
            if (numStems < 3)
                consensusAccuracyThresh = 0.60;
            hasStem = (meanStemAcc > 0.5)
                    && (meanAdapterAcc - meanStemAcc < 0.2)
                    && (consensusAccuracy >= consensusAccuracyThresh)
                    && (consensusAlnLength >= (0.65 * stemLength));

            if (hasStem)
            {
                // Iterate over the acceptedCells, modifying them
                for (size_t i = 0; i < acceptedCells.size(); ++i)
                {
                    auto& cell = acceptedCells[i];
                    const auto& preFlank = std::get<1>(precedingStemAlignments[i]);
                    const auto& postFlank = std::get<1>(postcedingStemAlignments[i]);
                    cell.score += preFlank.score;
                    cell.score += postFlank.score;
                    cell.matches += preFlank.matches + postFlank.matches;
                    cell.mismatches += preFlank.mismatches + postFlank.mismatches;
                    cell.insertions += preFlank.insertions + postFlank.insertions;
                    cell.deletions += preFlank.deletions + postFlank.deletions;
                    cell.readBases += preFlank.readBases + postFlank.readBases;
                    cell.jBeginPosition = cell.jBeginPosition - preFlank.mismatches
                                        - preFlank.matches - preFlank.insertions;
                    cell.jEndPosition = cell.jEndPosition + postFlank.mismatches
                                        + postFlank.matches + postFlank.insertions;
                }
            }
        }

        for (const auto& cell : acceptedCells)
        {
            // Create adapter label
            output.emplace_back(
                cell.jBeginPosition + extHQRegion.begin,
                cell.jEndPosition + extHQRegion.begin,
                cell.score, adapterNumber, RegionLabelType::ADAPTER);
            output.back().accuracy = Accuracy(cell);
            output.back().flankingScore = cell.flankingScore;
            output.back().hasStem = hasStem;
        }

        return output;
    }

    // This is currently used for DAFT
    std::vector<RegionLabel> LabelSingleAdapterTrimToLoop(
            const std::string& bases,
            const size_t adapterNumber,
            const RegionLabel& extHQRegion,
            const int minAdapterSpacing,
            const int stemLengthOverride = -1) const
    {
        std::vector<RegionLabel> output;
        bool hasStem = false;

        const int stemLength = (stemLengthOverride >= 0) ? stemLengthOverride : stemLength_;

        // stemLength_ is 0 by default, so this is a no-op when not looking for
        // loop
        const auto& adapter = adapterList_->at(adapterNumber).sequence;

        const auto& adapterStem = [&fullAdapter = adapterList_->at(adapterNumber).sequence,
                                   &stemLength]()
        {
            return fullAdapter.substr(fullAdapter.size() - stemLength, stemLength);
        }();

        // adapter length
        const auto adapterLength = adapter.size();
        const auto& hqRegionBases = bases.substr(extHQRegion.begin,
                                                 extHQRegion.end - extHQRegion.begin);

        // Fill matrix
        std::unique_ptr<int32_t[]> matrix{PairwiseAligner::SWComputeMatrix(
            adapter, hqRegionBases, adapterFindingScoring_, stemLength)};

        // DP constants
        const int32_t M = adapterLength + 1;
        const int32_t N = hqRegionBases.size() + 1;
        const int32_t beginLastRow = (M - 1) * N;

        float meanLoopAcc = 0;
        std::vector<Cell> loopCells;
        const int32_t beginLastLoopRow = (M - stemLength - 1) * N;

        // Convert last loop row to pairs of position/score
        std::vector<Cell> lastLoopRow;
        lastLoopRow.reserve(N);
        for (int j = 0; j < N; ++j)
        {
            // Filter by adapter score
            if (matrix[beginLastLoopRow + j] > minAdapterScore_)
                lastLoopRow.emplace_back(j, matrix[beginLastLoopRow + j]);
        }
        std::vector<Cell> rejectedLoops;
        if (!lastLoopRow.empty())
        {
            // Find possible adapter end positions with minimal spacing
            auto cells = SpacedSelector::FindEndPositions(
                std::move(lastLoopRow), minAdapterSpacing);

            const int loopLength = adapter.size() - 2 * stemLength;
            for (auto& cell : cells)
            {
                // Because this isn't a true local alignment backtrack, the
                // score will be diminished by the initial stem sequence if
                // that stem isn't present. We only use the accuracy,
                // however, and that will be unaffected.

                // Perform backtracking
                std::string stemLoop = adapter.substr(0, adapter.size() - stemLength);
                PairwiseAligner::SWBacktracking(matrix.get(), cell, N, stemLoop, hqRegionBases,
                                                adapterFindingScoring_, stemLength);

                // Compute accuracy
                float acc = Accuracy(cell);
#ifdef DEBUG_TOOLS
                // In debugMode, calculate and record all flanking scores
                PairwiseAligner::FlankingScore(
                    hqRegionBases, cell, flankingRegionsScoring_,
                    localAlnFlanking_, flankLength_);
                loopCells.push_back(cell);
                meanLoopAcc += acc;
#else
#ifdef DIAGNOSTICS
                // In DIAGNOSTICS mode, calculate all flanking scores if
                // they're being emitted. They will be recalculated if
                // necessary later on, so don't emitAdapterMetrics when
                // timing is critical
                if (emitAdapterMetrics_)
                    PairwiseAligner::FlankingScore(
                        hqRegionBases, cell, flankingRegionsScoring_,
                        localAlnFlanking_, flankLength_);
#endif

                if (acc < minHardAccuracy_) continue;
                if (acc >= ScaledMinSoftAccuracy(loopLength)
                    || (PairwiseAligner::FlankingScore(
                            hqRegionBases, cell, flankingRegionsScoring_,
                            localAlnFlanking_, flankLength_),
                        cell.flankingScore >= ScaledMinFlankingScore(loopLength)))
                {
                    loopCells.push_back(cell);
                    meanLoopAcc += acc;
                }
                else
                {
                    rejectedLoops.push_back(cell);
                }
#endif
            }
        }

        // Convert last row to pairs of position/score
        std::vector<Cell> lastRow;
        lastRow.reserve(N);
        for (int j = 0; j < N; ++j)
        {
            // Filter by adapter score
            if (matrix[beginLastRow + j] > minAdapterScore_)
                lastRow.emplace_back(j, matrix[beginLastRow + j]);
        }

        float meanAdapterAcc = 0.0;
        std::vector<Cell> acceptedCells;
        // Only continue if there are scores above the threshold
        if (!lastRow.empty())
        {
            // Find possible adapter end positions with minimal spacing
            auto cells = SpacedSelector::FindEndPositions(
                std::move(lastRow), minAdapterSpacing);

            size_t loopSearchIndex = 0;
            // For each possible adapter end position
            for (auto& cell : cells)
            {
                // Perform backtracking
                PairwiseAligner::SWBacktracking(matrix.get(), cell, N, adapter, hqRegionBases,
                                                adapterFindingScoring_);

                // Compute accuracy
                float acc = Accuracy(cell);

#ifdef DEBUG_TOOLS
                // In debugMode, calculate and record all flanking scores
                PairwiseAligner::FlankingScore(
                    hqRegionBases, cell, flankingRegionsScoring_,
                    localAlnFlanking_, flankLength_);

                acceptedCells.push_back(cell);
#else
#ifdef DIAGNOSTICS
                // In DIAGNOSTICS mode, calculate all flanking scores if
                // they're being emitted. They will be recalculated if
                // necessary later on, so don't emitAdapterMetrics when
                // timing is critical
                if (emitAdapterMetrics_)
                    PairwiseAligner::FlankingScore(
                        hqRegionBases, cell, flankingRegionsScoring_,
                        localAlnFlanking_, flankLength_);
#endif

                // see if there was a companion loop candidate:
                while (loopSearchIndex < loopCells.size()
                        && loopCells[loopSearchIndex].jEndPosition < cell.jBeginPosition)
                {
                    loopSearchIndex++;
                }
                // if there is, this cell passes
                if (loopSearchIndex < loopCells.size()
                        && loopCells[loopSearchIndex].jEndPosition > cell.jBeginPosition
                        && loopCells[loopSearchIndex].jBeginPosition < cell.jEndPosition)
                {
                    acceptedCells.push_back(cell);
                    meanAdapterAcc += acc;
                    continue;
                }

                // Otherwise only calculate the flanking score if needed
                // Hard filtering
                if (acc < minHardAccuracy_) continue;


                // Soft filtering
                if (acc >= ScaledMinSoftAccuracy(adapterLength)
                    || (PairwiseAligner::FlankingScore(
                            hqRegionBases, cell, flankingRegionsScoring_,
                            localAlnFlanking_, flankLength_),
                        cell.flankingScore >= ScaledMinFlankingScore(adapterLength)))
                {
                    acceptedCells.push_back(cell);
                    meanAdapterAcc += acc;
                }
#endif
            }
        }

        // Classify steminess
        if (!loopCells.empty() || !acceptedCells.empty())
        {
            // match, mismatch, gap open, gap extend
            auto alignmentEngine = spoa::createAlignmentEngine(
                spoa::AlignmentType::kSW, 4, 0, -7);
            auto graph = spoa::createGraph();

            // 10% margin for inserts in the stem and missed loop bases
            int searchBases = static_cast<int>(ceil(stemLength * 1.1));
            // the strings in the tuples are the raw candidate flanks
            std::vector<std::tuple<std::string, Cell>> precedingStemAlignments;
            std::vector<std::tuple<std::string, Cell>> postcedingStemAlignments;
            int consStemCount = 0;
            int searchWithin = (loopCells.size() >= acceptedCells.size())
                              ? 0 : stemLength_;
            const auto& toSearch = (loopCells.size() >= acceptedCells.size())
                                 ? loopCells : acceptedCells;

            for (const auto& cell : toSearch)
            {
                const auto& preStemFlank = SequenceUtilities::ReverseCompl(
                        GetPreFlank(hqRegionBases, cell, searchBases, true, searchWithin));
                const auto& postStemFlank = GetPostFlank(
                        hqRegionBases, cell, searchBases, true, searchWithin);
                precedingStemAlignments.push_back(
                    std::make_tuple(preStemFlank,
                                    AlignSuspectedStem(adapterStem, preStemFlank)));
                postcedingStemAlignments.push_back(
                    std::make_tuple(postStemFlank,
                                    AlignSuspectedStem(adapterStem, postStemFlank)));

                if (std::get<1>(precedingStemAlignments.back()).score > 8)
                {
                    auto preAlignment = alignmentEngine->align(
                        preStemFlank, graph);
                    graph->add_alignment(preAlignment, preStemFlank);
                    ++consStemCount;
                }
                if (std::get<1>(postcedingStemAlignments.back()).score > 8)
                {
                    auto postAlignment = alignmentEngine->align(
                        postStemFlank, graph);
                    graph->add_alignment(postAlignment, postStemFlank);
                    ++consStemCount;
                }
            }

            // Iterate over the outputs, classify the ZMW
            float meanStemAcc = 0;
            //float meanStemLen = 0;

            for (size_t i = 0; i < precedingStemAlignments.size(); ++i)
            {
                meanStemAcc += Accuracy(std::get<1>(precedingStemAlignments[i]));
                //meanStemLen += ReadLength(std::get<1>(precedingStemAlignments[i]));
                meanStemAcc += Accuracy(std::get<1>(postcedingStemAlignments[i]));
                //meanStemLen += ReadLength(std::get<1>(postcedingStemAlignments[i]));
            }
            auto numStems = precedingStemAlignments.size() + postcedingStemAlignments.size();
            float consensusAccuracy = 0.0;
            int consensusAlnLength = 0;
            // Only participate in consensus if we have enough observations
            if (consStemCount > 2 && consStemCount > 0.25 * loopCells.size())
            {
                auto consensus = graph->generate_consensus();
                auto consensusAlignment = AlignSuspectedStem(adapterStem, consensus);
                consensusAccuracy = Accuracy(consensusAlignment);
                consensusAlnLength = ReadLength(consensusAlignment);
            }
            if (numStems)
            {
                meanStemAcc /= numStems;
                //meanStemLen /= numStems;
            }
            if (loopCells.size())
                meanLoopAcc /= loopCells.size();
            if (acceptedCells.size())
                meanAdapterAcc /= acceptedCells.size();

            float consensusAccuracyThresh = 0.80;
            float consFraction = static_cast<float>(consStemCount) / toSearch.size();
            // there should be two consensus observations per adapter:
            consFraction /= 2.0f;
            // If there are 1 or 2 adapters, with 1 or 2 observations, be a
            // bit more lenient

            // A simple and somewhat crude decision tree. There are some
            // additional constants here that can be plumbed out and trained if
            // necessary
            if (consStemCount < 3 && toSearch.size() < 3)
                consensusAccuracyThresh = 0.60;
            hasStem = (meanStemAcc > 0.75) ? true : false;
            hasStem = ((consensusAccuracy >= consensusAccuracyThresh)
                        && (consensusAlnLength >= (0.65 * stemLength))
                        && consFraction > 0.5) ? true : hasStem;
            hasStem = (consStemCount < 3 && meanAdapterAcc - meanLoopAcc > 0.02
                       && (meanStemAcc > 0.5 || consensusAccuracy > 0.5)) ? true : hasStem;
            hasStem = (loopCells.size() > 0 && meanAdapterAcc - meanLoopAcc > 0.08) ? true : hasStem;
            hasStem = (meanLoopAcc - meanAdapterAcc > 0.25) ? false : hasStem;
            hasStem = (loopCells.size() > acceptedCells.size() * 2) ? false : hasStem;
            hasStem = (acceptedCells.size() > loopCells.size() * 2
                       && (meanStemAcc > 0.5 && (toSearch.size() < 3 || consensusAccuracy > 0.5)))
                       ? true : hasStem;

            // if loopCells is empty, what should we do? Dumpster dive!
            // TODO: if loopCells is just a heck of a lot shorter? Can we
            // dumpster dive then too?
            size_t loopI = 0;
            if (!hasStem && loopCells.empty() && !rejectedLoops.empty())
            {
                for (const auto& fullCell : acceptedCells)
                {
                    while (loopI < loopCells.size()
                            && rejectedLoops[loopI].jEndPosition < fullCell.jBeginPosition)
                    {
                        loopI++;
                    }
                    if (loopI < rejectedLoops.size()
                            && rejectedLoops[loopI].jEndPosition > fullCell.jBeginPosition
                            && rejectedLoops[loopI].jBeginPosition < fullCell.jEndPosition)
                    {
                        loopCells.push_back(rejectedLoops[loopI]);
                    }
                }
            }
            // we couldn't recycle, admit defeat:
            if (loopCells.empty() && !acceptedCells.empty())
            {
                hasStem = true;
            }
            if (!hasStem)
            {
                acceptedCells = loopCells;
            }
        }
        else if (!acceptedCells.empty())
            hasStem = true;

        // after this point, acceptedCells can be loop or full candidates
        for (const auto& cell : acceptedCells)
        {
            // Create adapter label
            output.emplace_back(
                cell.jBeginPosition + extHQRegion.begin,
                cell.jEndPosition + extHQRegion.begin,
                cell.score, adapterNumber, RegionLabelType::ADAPTER);
            output.back().accuracy = Accuracy(cell);
            output.back().flankingScore = cell.flankingScore;
            output.back().hasStem = hasStem;
        }

        return output;
    }

};

}}} // ::PacBio::Primary::Postprimary
