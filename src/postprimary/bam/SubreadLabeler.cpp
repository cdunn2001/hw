// Copyright (c) 2014-2018, Pacific Biosciences of California, Inc.
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

#include <pacbio/logging/Logger.h>

#include <bazio/BlockLevelMetrics.h>
#include <bazio/FastaUtilities.h>
#include <bazio/SmartMemory.h>

#include <postprimary/adapterfinder/AdapterCorrector.h>
#include <postprimary/stats/ZmwMetrics.h>
#include <postprimary/stats/ProductivityClass.h>

#include "SubreadLabeler.h"
#include "SubreadSplitter.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {


namespace {

std::vector<ResultPacket> CollectScraps(const std::vector<RegionLabel>& scraps,
                                        const RegionLabel& hqregion,
                                        const ProductivityInfo& pinfo,
                                        const EventData& events,
                                        const RegionLabelType& label,
                                        bool nobam,
                                        std::string movieName,
                                        const std::unordered_map<size_t, size_t>& base2frame)
{
    std::vector<ResultPacket> resultVec;
    resultVec.reserve(scraps.size());
    // For all labels within a HQregion
    for (const auto& scrap : scraps)
    {
        if (scrap.begin < hqregion.end && scrap.end > hqregion.begin)
        {
            // Determine correct begin and end of scraps
            int begin = std::max(scrap.begin, hqregion.begin);
            int end   = std::min(scrap.end,   hqregion.end);
            // Create ResultPacket
            auto result = SubreadSplitter::PartialReadToResultPacket(
                pinfo, events, begin, end, label, nobam, movieName, base2frame);
            resultVec.emplace_back(std::move(result));
        }
    }
    return resultVec;
}

std::vector<SubreadContext> SubreadsByAdapters(const std::vector<RegionLabel>& adapters,
                                               uint32_t numBases)
{
    std::vector<SubreadContext> subReadBoundaries;
    subReadBoundaries.reserve(adapters.size() + 1);
    int begin = 0;
    for (size_t i = 0; i < adapters.size(); ++i)
    {
        const auto& adapter = adapters[i];
        if (adapter.begin - begin > 0)
        {
            // Compute subread boundaries
            SubreadContext lc(begin, adapter.begin);
            lc.adapterAfter = true;
            lc.adapterIdAfter = adapter.sequenceId;
            lc.adapterFlankingScoreAfter = adapter.flankingScore;
            lc.adapterAccuracyAfter = adapter.accuracy;
            lc.hasStemAfter = adapter.hasStem;
            if (begin != 0)
            {
                lc.adapterBefore = true;
                if (i > 0) {
                    lc.adapterIdBefore = adapters[i - 1].sequenceId;
                    lc.adapterFlankingScoreBefore = adapters[i - 1].flankingScore;
                    lc.adapterAccuracyBefore = adapters[i - 1].accuracy;
                    lc.hasStemBefore = adapters[i - 1].hasStem;
                }
            }
            else
            {
                lc.adapterBefore = false;
            }
            subReadBoundaries.emplace_back(std::move(lc));
        }
        begin = adapter.end;
    }
    SubreadContext lcLastSubread(begin, numBases);
    lcLastSubread.adapterAfter = false;
    lcLastSubread.adapterBefore = true;
    lcLastSubread.adapterIdBefore = adapters[adapters.size() - 1].sequenceId;
    lcLastSubread.adapterFlankingScoreBefore = adapters[adapters.size() - 1].flankingScore;
    lcLastSubread.adapterAccuracyBefore = adapters[adapters.size() - 1].accuracy;
    lcLastSubread.hasStemBefore = adapters[adapters.size() - 1].hasStem;
    subReadBoundaries.emplace_back(std::move(lcLastSubread));

    return subReadBoundaries;
}

void ShrinkSubreadsByBarcodes(const std::vector<RegionLabel>& barcodes,
                              std::vector<SubreadContext>* subReadBoundaries)
{
    for (const auto& barcode : barcodes)
    {
        // Find subread boundaries that are close to a barcode
        // and extend boundaries
        auto sub = subReadBoundaries->begin();
        while (sub != subReadBoundaries->end())
        {
            if (sub->end == barcode.end)
            {
                sub->end = barcode.begin;
                sub->barcodeIdAfter = barcode.sequenceId;
            }
            if (barcode.begin == sub->begin)
            {
                sub->begin = barcode.end;
                sub->barcodeIdBefore = barcode.sequenceId;
            }
            // If subread is negative, due to missing insert, remove sub
            if (sub->end < sub->begin)
                subReadBoundaries->erase(sub);
            else
                ++sub;
        }
    }
}

std::vector<ResultPacket> CollectLQRegions(const ProductivityInfo& pinfo,
                                           const EventData& events,
                                           const RegionLabel& hqregion,
                                           uint32_t numBases,
                                           bool nobam,
                                           std::string movieName,
                                           const std::unordered_map<size_t, size_t>& base2frame)
{
    std::vector<ResultPacket> lqregions;
    uint64_t begin = 0;
    // Only export if LQRegion has bases
    if (hqregion.begin - begin > 0)
    {
        auto newRead = SubreadSplitter::PartialReadToResultPacket(
            pinfo, events, begin, hqregion.begin, RegionLabelType::LQREGION, nobam, movieName, base2frame);
        if (newRead.length > 0)
            lqregions.push_back(std::move(newRead));
    }
    begin = hqregion.end;
    if (begin != numBases)
    {
        auto newRead = SubreadSplitter::PartialReadToResultPacket(
            pinfo, events, begin, numBases, RegionLabelType::LQREGION, nobam, movieName, base2frame);
        if (newRead.length > 0)
            lqregions.push_back(std::move(newRead));
    }

    return lqregions;
}

void DefineSubreadOrientation(std::vector<ResultPacket>* resultVec,
                              const double barcodeScore,
                              bool cxTagAlreadyCalled,
                              bool emitAdapterMetrics,
                              BarcodeStrategy scoreMode)
{
    // Orientation can only be defined if asymmentric barcodes are present
    // OR if asymmetric adapters are available.
    std::set<int> adapterIDs;
    // Collect all adapter ids
    for (const auto& r : *resultVec)
    {
        if (r.context.label == RegionLabelType::INSERT)
        {
            if (r.context.adapterIdBefore != SubreadContext::UNSET)
                adapterIDs.insert(r.context.adapterIdBefore);
            if (r.context.adapterIdAfter != SubreadContext::UNSET)
                adapterIDs.insert(r.context.adapterIdAfter);
        }
    }
    bool asymmetricAdapters = false;
    if (adapterIDs.size() > 2) 
        throw std::runtime_error("Not aware of a scenario with three adapters");
    else if (adapterIDs.size() == 2 && !cxTagAlreadyCalled)
    {
        asymmetricAdapters = true;
        // Forward direction is defined as first adapter left and / or 
        // second adapter right
        auto it = adapterIDs.begin();
        int left  = *it++;
        int right = *it;
        for (auto& r : *resultVec)
        {
            if (r.context.label == RegionLabelType::INSERT)
            {
                uint8_t oldTag = boost::numeric_cast<uint8_t>(r.cx);
                if (r.context.adapterIdBefore == left
                    || r.context.adapterIdAfter == right)
                   r.tags["cx"] = boost::numeric_cast<uint8_t>(oldTag | 16);
                else if (r.context.adapterIdBefore == right
                    || r.context.adapterIdAfter == left)
                   r.tags["cx"] = boost::numeric_cast<uint8_t>(oldTag | 32);
            }
        }
    }
#ifdef DIAGNOSTICS
    // Adapter Details "ad" BAM tag:
    //      ad:Z:<adapterBeforeDetails>;<adapterAfterDetails>
    // where <adapter{Before/After}Details> is
    // "<adapterId>,<adapterAccuracy>,<adapterFlankingScore>,<hasStem>".
    //
    // If an adapter is missing, the details are ".".
    if (emitAdapterMetrics)
    {
        for (auto& r: *resultVec)
        {
            if (r.context.label == RegionLabelType::INSERT)
            {
                std::ostringstream adapterDetails;
                adapterDetails.precision(4);
                if (r.context.adapterIdBefore == -1)
                {
                    adapterDetails << ".";
                }
                else
                {
                    adapterDetails
                        << r.context.adapterIdBefore
                        << "," << r.context.adapterAccuracyBefore
                        << "," << r.context.adapterFlankingScoreBefore
                        << "," << r.context.hasStemBefore;
                }
                adapterDetails << ";";
                if (r.context.adapterIdAfter == -1)
                {
                    adapterDetails << ".";
                }
                else
                {
                    adapterDetails
                        << r.context.adapterIdAfter
                        << "," << r.context.adapterAccuracyAfter
                        << "," << r.context.adapterFlankingScoreAfter
                        << "," << r.context.hasStemAfter;
                }
                r.tags["ad"] = adapterDetails.str();
            }
        }
    }
#endif
    // Barcodes
    std::map<std::pair<int, int>, std::vector<SubreadContext>> bcPairToContext;
    int bqValue = static_cast<int>(round(barcodeScore));
    // Collect all barcode sets, adapter preceding and successive,
    // and save in a std::map to avoid duplicates.
    for (const auto& r : *resultVec)
    {
        // We only need to find the BarcodePair for results that are
        //  (A) Inserts and (B) have at least one set barcode label
        if (r.context.label == RegionLabelType::INSERT
            && (r.context.barcodeIdBefore != SubreadContext::UNSET ||
                r.context.barcodeIdAfter  != SubreadContext::UNSET))
        {
            // Create a pair to store the result
            std::pair<int, int> bcPair;

            // Since we know all possible pairs, and that there are no 
            //  overlaps we only need to know 1 barcode Id
            const int barcodeId = (r.context.barcodeIdBefore != SubreadContext::UNSET)
                    ? r.context.barcodeIdBefore
                    : r.context.barcodeIdAfter;

            if (scoreMode == BarcodeStrategy::ASYMMETRIC)  // Asymmetric barcoding
            {
                // In Asymmetric barcoing, the bcPair is an odd-even pair 
                //  of adjacent integers, e.g. (36,37).
                if (barcodeId % 2 == 0)   
                    bcPair = std::make_pair(barcodeId, barcodeId + 1);
                else
                    bcPair = std::make_pair(barcodeId - 1, barcodeId);
            }
            else  // Symmetric barcoding
            {
                bcPair = std::make_pair(barcodeId, barcodeId);
            }

            // If there is no barcode pair stored with the same ids
            if (bcPairToContext.find(bcPair) == bcPairToContext.end())
                bcPairToContext[bcPair] = std::vector<SubreadContext>();

            // Collect SubreadContexts w.r.t. barcode pair
            bcPairToContext[bcPair].push_back(r.context);
        }
    }

    // Remove fake barcode calls with identical barcodes, in asymmetric mode.
    // This is not a problem for symmetric barcodes, as we will have only
    // one pair with identical barcodes.
    if (bcPairToContext.size() > 1)
    {
        for (auto it = bcPairToContext.cbegin(); it != bcPairToContext.cend();)
        {
            if (it->first.first == it->first.second)
                bcPairToContext.erase(it++);
            else
                ++it;
        }
    }
    if (bcPairToContext.size() == 1)
    {
        // Subreads with a preceding barcode 'a' are forward
        uint16_t a = boost::numeric_cast<uint16_t>(bcPairToContext.begin()->first.first);
        uint16_t b = boost::numeric_cast<uint16_t>(bcPairToContext.begin()->first.second);

        std::vector<uint16_t> bc{a, b};

        bool forward = true;
        for (auto& r : *resultVec)
        {
            if (r.context.label == RegionLabelType::INSERT)
            {
                // Determine subread direction iff barcodes are asymmetric,
                // adapters are symmetric, and the cxTag has not been
                // provided by bam2bam
                if (scoreMode == BarcodeStrategy::ASYMMETRIC
                    && !asymmetricAdapters && !cxTagAlreadyCalled)
                {
                    uint8_t oldTag = static_cast<uint8_t>(r.cx);

                    // Do not determine direction if only one barcode is present
                    if (r.context.barcodeIdBefore == SubreadContext::UNSET &&
                        r.context.barcodeIdAfter == SubreadContext::UNSET)
                    {
                        if (forward)
                            r.tags["cx"] = static_cast<uint8_t>(oldTag | 32);
                        else
                            r.tags["cx"] = static_cast<uint8_t>(oldTag | 16);
                        forward ^= true;
                    }
                    else if (r.context.barcodeIdBefore == a)
                        r.tags["cx"] = static_cast<uint8_t>(oldTag | 16);
                    else
                        r.tags["cx"] = static_cast<uint8_t>(oldTag | 32);
                }

                r.tags["bc"] = bc;
                r.tags["bq"] = bqValue;
            }
        }
    }
    else
    {
        // std::cerr << "Did not find correct number of adapter pairs "
        //           << barcodes.size() << std::endl;
    }
}

void ApplyCxTags(std::vector<ResultPacket>* resultVec,
                 const std::vector<uint8_t>& cxTags)
{
    for (auto& r : *resultVec)
    {
        switch(r.context.label)
        {
            case RegionLabelType::INSERT:
                if (r.cxTagIdx >= 0)
                {
                    if (!r.skipExistingCxTag)
                    {
                        r.tags["cx"] = cxTags[r.cxTagIdx];
                        r.cx = cxTags[r.cxTagIdx];
                    }
                }
                else
                {
                    if (r.cxBitsToPreserve)
                    {
                        r.tags["cx"] = r.tags["cx"].ToUInt8() | r.cxBitsToPreserve;
                        r.cx |= r.cxBitsToPreserve;
                    }
                }
                break;
            default: break;
        }
    }
}

void CheckForExistingCxTag(ResultPacket& newRead,
                           const SubreadContext& sub,
                           const std::vector<RegionLabel>& inserts,
                           const std::vector<RegionLabel>& filtered,
                           const std::vector<RegionLabel>& lqs,
                           const std::vector<uint8_t>& cxTags)
{
    bool existingFilter = false;

    // Check if read had a cx tag to begin with.
    for (size_t insertNum = 0; insertNum < inserts.size(); insertNum++)
    {
        const auto& insert = inserts[insertNum];
        const auto& cxTag = cxTags[insertNum];

        if (insert.begin == sub.begin && insert.end == sub.end)
        {
            // Matches exactly to existing insert, no need to recompute cx tag.
            newRead.cxTagIdx = insertNum;
            break;
        }
        else if (insert.begin == sub.begin)
        {
            // Overlaps with upstream insert coordinate,
            // check cx tag for upstream bits.
            uint8_t cx = 0;
            if ((cxTag >> 0) & 1) cx |= 1;
            if ((cxTag >> 2) & 1) cx |= 4;
            if ((cxTag >> 6) & 1) cx |= 64;
            newRead.cxBitsToPreserve = cx;
        }
        else if (insert.end == sub.end)
        {
           // Overlaps with downstream insert coordinate,
           // check cx tag for downstream bits.
            uint8_t cx = 0;
            if ((cxTag >> 1) & 1) cx |= 2;
            if ((cxTag >> 3) & 1) cx |= 8;
            if ((cxTag >> 7) & 1) cx |= 128;
            newRead.cxBitsToPreserve = cx;
        }

        if (newRead.cxBitsToPreserve)
        {
            if ((cxTag >> 4) & 1) newRead.cxBitsToPreserve |= 16;
            if ((cxTag >> 5) & 1) newRead.cxBitsToPreserve |= 32;
        }

        // Check if read overlaps with filtered region.
        for (const auto filter : filtered)
        {
            if (std::max(filter.begin, sub.begin) < std::min(filter.end, sub.end))
            {
                existingFilter = true;
                break;
            }
        }

        // Check if read overlaps with low-quality region.
        for (const auto lq : lqs)
        {
            if (std::max(lq.begin, sub.begin) < std::min(lq.end, sub.end))
            {
                existingFilter = true;
                break;
            }
        }

    }

    // If the read now overlaps with a filtered or low-quality region,
    // we want to skip the cx tag that would have been associated with
    // this read since we want this information to be recomputed.
    if (existingFilter)
    {
        // Only skip tag if it previously had one.
        if (newRead.cxTagIdx >= 0) newRead.skipExistingCxTag = true;
    }
}


} // anonymous

SubreadLabeler::SubreadLabeler(const std::shared_ptr<UserParameters>& user,
                               const std::shared_ptr<RuntimeMetaData>& rmd,
                               const std::shared_ptr<PpaAlgoConfig>& ppaAlgoConfig)
    : user_(user)
    , rmd_(rmd)
    , ppaAlgoConfig_(ppaAlgoConfig)
{
    InitFromPpaAlgoConfig();
    InitAdapterFinding();
    InitControlFiltering();
}

ControlMetrics SubreadLabeler::CallControls(
    const ProductivityInfo& pinfo, const EventData& events,
    const RegionLabel& hqregion, bool isControl) const
{
    ControlMetrics controlMetrics;
    controlMetrics.isControl = isControl;

    // Run spike-in control analysis up front on P1 reads.
    // We check the control flag for the case of running bam2bam
    // where reads will have the control flag set if annotated as such
    // in the BAM files. This leaves control reads untouched and no
    // further analysis is done unless we are asked to perform
    // control filtering again.

    // Minimal polymerase read length check, based on #bases
    // Only run control filtering if requested.
    if (pinfo.productivity == ProductivityClass::PRODUCTIVE
            && performControlFiltering_
            && !(events.NumBases() < user_->minPolymerasereadLength))
    {
        controlMetrics = FilterControls(hqregion, events);
    }

    return controlMetrics;
}

std::tuple<std::vector<RegionLabel>, AdapterMetrics, RegionLabel>
SubreadLabeler::CallAdapters(const EventData& events,
    const RegionLabel& hqregion, bool isControl, bool emptyPulseCoordinates) const
{
    AdapterMetrics adapterMetrics;
    RegionLabel modifiedHqregion(hqregion);
    bool hasStem = false;

    std::vector<RegionLabel> adapters;
    // Minimal polymerase read length check, based on #bases
    // and check if we actually have an HQ-region. The
    // flag emptyPulseCoordinates is needed since for a bam2bam
    // workflow these values will not be set.
    if (!isControl && !(events.NumBases() < user_->minPolymerasereadLength)
                   && performAdapterFinding_
                   && (!(hqregion.pulseBegin == 0 && hqregion.pulseEnd == 0) || emptyPulseCoordinates))
    {
        std::string bases(events.BaseCalls().begin(),
            events.BaseCalls().end());

        // Label adapter
        adapterMetrics.called = true;

        adapters = adapterLabeler_->Label(
            bases, modifiedHqregion);

        for (const auto& adp : adapters)
            if (adp.hasStem)
                hasStem = true;

        adapterMetrics.numFound = adapters.size();

        if (user_->correctAdapters)
        {
            adapterMetrics.corrected = true;
            // We only want to use it once:
            const auto& flagged = FlagSubreadLengthOutliers(
                adapters, modifiedHqregion);
            if (user_->sensitiveCorrector)
            {
                adapters = SensitiveAdapterSearch(
                    bases, adapters, modifiedHqregion, flagged,
                    *sensitiveAdapterLabeler_,
                    *rcAdapterLabeler_);
            }
            adapterMetrics.numFoundSensitive = static_cast<int32_t>(adapters.size())
                - adapterMetrics.numFound;

            const auto& alsoFlagged = FlagAbsentAdapters(
                bases, adapters, *palindromeScorer_);
            if (alsoFlagged.size() > 0
                    || adapterMetrics.numFoundSensitive < static_cast<int32_t>(flagged.size()))
            {
                adapters = PalindromeSearch(bases, adapters, modifiedHqregion);
                adapterMetrics.numFoundPalindrome = static_cast<int32_t>(adapters.size())
                    - adapterMetrics.numFound
                    - adapterMetrics.numFoundSensitive;
                if (adapterMetrics.numFoundPalindrome < 0)
                {
                    adapterMetrics.numRemoved = 0 - adapterMetrics.numFoundPalindrome;
                    adapterMetrics.numFoundPalindrome = 0;
                }
            }
            else
            {
                // If we didn't run palindrome finder, which does its own FP
                // removal, try running a standalone remover:
                auto numBefore = adapters.size();
                adapters = RemoveFalsePositives(bases, adapters, modifiedHqregion);
                adapterMetrics.numRemoved += static_cast<int32_t>(numBefore - adapters.size());
            }
        }
        // The subread splitting gets quite messed up if adapters are unsorted.
        std::sort(adapters.begin(), adapters.end(),
                  [](const RegionLabel& l1, const RegionLabel& l2)
            { return std::tie(l1.begin, l1.end) < std::tie(l2.begin, l2.end); });
    }

    // hasStem is determined on a per-ZMW basis. Every adapter in a ZMW will
    // have the same hasStem value.
    if (user_->lookForLoop || user_->trimToLoop)
    {
        if ((adapters.size() > 0 && hasStem) || isControl)
        {
            adapterMetrics.hasStem = true;
        }
        else
        {
            adapterMetrics.hasStem = false;
        }
    }
    else
    {
        adapterMetrics.hasStem = true;
    }

    return std::make_tuple(std::move(adapters), adapterMetrics,
                           std::move(modifiedHqregion));
}

std::vector<ResultPacket> SubreadLabeler::ReadToResultPacket(
    const ProductivityInfo& pinfo, const EventData& events,
    const RegionLabels& regions, const double barcodeScore,
    const ControlMetrics& controlMetrics) const
{
    //FYI: Boundaries are [)
    //R -------------------------------------------------------------
    //A               b   e                   b   e
    //B             b e   b e               b e   b  e
    //
    //
    //S b             e   b                   e   b                 e
    //S b           e       b               e        b              e


    const auto& hqregion = regions.hqregion;
    const auto& adapters = regions.adapters;
    const auto& barcodes = regions.barcodes;
    const auto& cxTags = regions.cxTags;
    const auto& filtered = regions.filtered;
    const auto& lqs = regions.lq;
    const auto& inserts = regions.inserts;

    std::vector<ResultPacket> resultVec;

    // Check if subread is long enough
    auto IsFiltered = [&pinfo, this](const int length, const int estInsertLength) {
        const bool isLongEnough = (length >= static_cast<int>(user_->minSubreadLength)
                                   && estInsertLength >= static_cast<int>(user_->minSubreadLength));
        assert(pinfo.productivity != ProductivityClass::EMPTY);
        if (isLongEnough && pinfo.productivity == ProductivityClass::PRODUCTIVE)
            return RegionLabelType::INSERT;
        else
            return RegionLabelType::FILTERED;
    };

    if (!(hqregion.begin == hqregion.end) && !user_->polymeraseread)
    {
        // If adapters and HQRegions are available,
        // create subread boundaries, and collect scraps
        if (!adapters.empty())
        {
            // unsorted adapters will cause the splitting algorithm below to
            // misbehave, so we'll check that they are infact sorted
            assert(std::is_sorted(adapters.begin(), adapters.end(),
                      [](const RegionLabel& l1, const RegionLabel& l2)
                { return std::tie(l1.begin, l1.end) < std::tie(l2.begin, l2.end); }));
            // Collect adapters as scraps
            MoveAppend(CollectScraps(adapters, hqregion, pinfo, events,
                                     RegionLabelType::ADAPTER,
                                     user_->nobam, rmd_->movieName, regions.base2frame),
                       resultVec);
            // Collect barcodes as scraps
            if (!barcodes.empty())
                MoveAppend(CollectScraps(barcodes, hqregion, pinfo, events,
                                         RegionLabelType::BARCODE,
                                         user_->nobam, rmd_->movieName, regions.base2frame),
                           resultVec);

            // Create subreads by adapter calls
            std::vector<SubreadContext> subReadBoundaries;
            MoveAppend(SubreadsByAdapters(adapters, events.NumBases()), subReadBoundaries);

            // If barcodes are available
            if (!barcodes.empty())
                ShrinkSubreadsByBarcodes(barcodes, &subReadBoundaries);

            // Filter out ZMWs with insert lengths below our minSubreadLength
            // thershold:
            auto estInsertLength = 0;
            if (!subReadBoundaries.empty())
            {
                std::vector<int> subLengths;
                subLengths.reserve(subReadBoundaries.size());
                for (const auto& sub : subReadBoundaries)
                {
                    subLengths.push_back(std::min(hqregion.end, sub.end) -
                                         std::max(hqregion.begin, sub.begin));
                }
                // Use highest quartile, so we're not throwing out cases with a
                // few short inserts (or partial reads!) and a few long inserts
                size_t quanti = subLengths.size() * 0.75;
                std::nth_element(subLengths.begin(),
                                 subLengths.begin() + quanti,
                                 subLengths.end());
                estInsertLength = subLengths[quanti];
            }

            // Export each subread that is in a HQ region
            for (auto sub : subReadBoundaries)
            {
                if (sub.begin < hqregion.end && sub.end > hqregion.begin)
                {
                    // Determine correct begin and end of inserts
                    // w.r.t. to HQ region
                    if (sub.begin <= hqregion.begin)
                    {
                        sub.adapterBefore = false;
                        sub.barcodeIdBefore = SubreadContext::UNSET;
                        sub.begin = hqregion.begin;
                    }
                    if (sub.end >= hqregion.end)
                    {
                        sub.adapterAfter = false;
                        sub.barcodeIdAfter = SubreadContext::UNSET;
                        sub.end = hqregion.end;
                    }

                    // Independent of existing filtering from bam2bam,
                    // treat subread as fresh and check if subread is long enough
                    sub.label = IsFiltered(sub.end - sub.begin, estInsertLength);

                    // Export
                    auto newRead = SubreadSplitter::PartialReadToResultPacket(
                            pinfo, events, sub, user_->nobam, rmd_->movieName, regions.base2frame);

                    if (!cxTags.empty())
                        CheckForExistingCxTag(newRead, sub, inserts, filtered, lqs, cxTags);

                    // Only add it if it has bases.
                    if (newRead.length > 0)
                        resultVec.push_back(std::move(newRead));
                }
            }

            // If there are existing cx tags created by bam2bam
            if (!cxTags.empty())
                ApplyCxTags(&resultVec, cxTags);

            // Determine subread orientation by asymmetric barcodes.
            DefineSubreadOrientation(&resultVec, barcodeScore, !cxTags.empty(),
                                     user_->emitAdapterMetrics, user_->scoreMode);
        }
        else
        {
            ResultPacket newRead;

            // Export this subread that is not defined by adapters, but
            // solely by the HQ-region
            if (!user_->hqonly)
                newRead = SubreadSplitter::PartialReadToResultPacket(
                    pinfo, events, SubreadContext(hqregion.begin, hqregion.end,
                    IsFiltered(hqregion.end - hqregion.begin, hqregion.end - hqregion.begin)), user_->nobam,
                    rmd_->movieName, regions.base2frame);
            // Export as a HQ-region
            else
                newRead = SubreadSplitter::PartialReadToResultPacket(
                    pinfo, events, hqregion.begin, hqregion.end,
                    RegionLabelType::INSERT, user_->nobam,
                    rmd_->movieName, regions.base2frame);

            if (!cxTags.empty())
                CheckForExistingCxTag(newRead, SubreadContext(hqregion.begin, hqregion.end), inserts, filtered, lqs, cxTags);

            if (newRead.length > 0)
                resultVec.push_back(std::move(newRead));
        }

        // If there are existing cx tags created by bam2bam
        if (!user_->hqonly)
        {
            if (!cxTags.empty())
                ApplyCxTags(&resultVec, cxTags);

            // Determine subread orientation by asymmetric barcodes.
            DefineSubreadOrientation(&resultVec, barcodeScore, !cxTags.empty(),
                                     user_->emitAdapterMetrics, user_->scoreMode);
        }

        // Collect LQRegions
        MoveAppend(CollectLQRegions(
                        pinfo, events, hqregion, events.NumBases(),
                        user_->nobam, rmd_->movieName, regions.base2frame),
                   resultVec);
    }
    // Export one polymeraseread
    else if (user_->polymeraseread)
    {
        // NOTE: We don't export the scraps (LQRegion and Filtered).
        auto poly = SubreadSplitter::PartialReadToResultPacket(
                        pinfo, events, SubreadContext(0, events.NumBases(),
                        RegionLabelType::POLYMERASEREAD),
                        user_->nobam,
                        rmd_->movieName,
                        regions.base2frame);
        if (poly.length > 0)
            resultVec.push_back(std::move(poly));
    }
    // Otherwise export HQ-regions
    else if (user_->hqonly)
    {
        if (hqregion.begin != hqregion.end)
        {
            auto newRead = SubreadSplitter::PartialReadToResultPacket(
                    pinfo, events, hqregion.begin, hqregion.end,
                    RegionLabelType::INSERT, user_->nobam,
                    rmd_->movieName,
                    regions.base2frame);
            if (newRead.length > 0)
                resultVec.push_back(std::move(newRead));
        }

        // Collect LQRegions
        MoveAppend(CollectLQRegions(pinfo, events, hqregion, events.NumBases(),
                                    user_->nobam, rmd_->movieName, regions.base2frame),
                   resultVec);
    }
    else
    {
        auto lqread = SubreadSplitter::PartialReadToResultPacket(
                        pinfo, events, SubreadContext(0, events.NumBases(),
                        RegionLabelType::LQREGION), user_->nobam,
                        rmd_->movieName, regions.base2frame);
        if (lqread.length > 0)
            resultVec.push_back(std::move(lqread));
    }

    // If read is a control, label all results as control.
    //  This is done outside FilterControls, to support already labeled
    //  reads from bam2bam in the sanity mode that forwards existing labels.
    if (controlMetrics.isControl)
    {
        for (auto& result : resultVec)
        {
            result.control = true;
#ifdef DIAGNOSTICS
            if (user_->emitControlMetrics &&
                performControlFiltering_ &&
                result.context.label == RegionLabelType::INSERT)
            {
                std::ostringstream controlDetails;
                controlDetails.precision(4);
                controlDetails << controlMetrics.controlReadLength << "," << controlMetrics.controlReadScore;
                result.tags["cm"] = controlDetails.str();
            }
#endif
        }
    }

    // Now that we have empty regions, we need to sort by first and last
    // position:
    std::sort(resultVec.begin(), resultVec.end(),
              [](const ResultPacket& l1, const ResultPacket& l2)
        { return std::tie(l1.startPos, l1.endPos) < std::tie(l2.startPos, l2.endPos); });

    return resultVec;
}

ControlMetrics SubreadLabeler::FilterControls(const RegionLabel& hqregion, const EventData& events) const
{
    ControlMetrics ret;
    ret.called = true;
    ret.isControl = false;
    if (!(hqregion.begin == hqregion.end))
    {
        std::string bases(events.BaseCalls().begin() + hqregion.begin,
                          events.BaseCalls().begin() + hqregion.end);
        ControlHit hit = controlFilter_->AlignToControl(bases);
        if (hit.accuracy > 0)
        {
            ret.isControl = true;
            ret.controlReadLength = hit.span;
            ret.controlReadScore = hit.accuracy / 100.0f;
        }
    }
    return ret;
}

void SubreadLabeler::InitFromPpaAlgoConfig()
{
    performAdapterFinding_  = !ppaAlgoConfig_->adapterFinding.disableAdapterFinding;
    performControlFiltering_ = !ppaAlgoConfig_->controlFilter.disableControlFiltering;

    if (performAdapterFinding_)
    {
        adapterList_ = std::make_shared<std::vector<FastaEntry>>();
        rcAdapterList_ = std::make_shared<std::vector<FastaEntry>>();

        if (ppaAlgoConfig_->adapterFinding.leftAdapter == ppaAlgoConfig_->adapterFinding.rightAdapter)
        {
            adapterList_->emplace_back("adapter", ppaAlgoConfig_->adapterFinding.leftAdapter, 0);
            rcAdapterList_->emplace_back("rcAdapter", SequenceUtilities::ReverseCompl(ppaAlgoConfig_->adapterFinding.leftAdapter), 256);
        }
        else
        {
            adapterList_->emplace_back("leftAdapter", ppaAlgoConfig_->adapterFinding.leftAdapter, 0);
            adapterList_->emplace_back("rightAdapter", ppaAlgoConfig_->adapterFinding.rightAdapter, 1);
            rcAdapterList_->emplace_back("rcLeftAdapter", SequenceUtilities::ReverseCompl(ppaAlgoConfig_->adapterFinding.leftAdapter), 256);
            rcAdapterList_->emplace_back("rcRightAdapter", SequenceUtilities::ReverseCompl(ppaAlgoConfig_->adapterFinding.rightAdapter), 257);
        }
    }

    if (performControlFiltering_)
    {
        controlList_ = std::make_shared<std::vector<FastaEntry>>();
        controlList_->emplace_back("control", ppaAlgoConfig_->controlFilter.sequence, 0);

        controlAdapterList_ = std::make_shared<std::vector<FastaEntry>>();

        if (ppaAlgoConfig_->controlFilter.leftAdapter == ppaAlgoConfig_->controlFilter.rightAdapter)
        {
            controlAdapterList_->emplace_back("adapter", ppaAlgoConfig_->controlFilter.leftAdapter, 0);
        }
        else
        {
            controlAdapterList_->emplace_back("leftAdapter", ppaAlgoConfig_->controlFilter.leftAdapter, 0);
            controlAdapterList_->emplace_back("rightAdapter", ppaAlgoConfig_->controlFilter.rightAdapter, 1);
        }
    }
}

void SubreadLabeler::InitAdapterFinding()
{
    if (performAdapterFinding_)
    {
        adapterLabeler_ =
                std::make_unique<AdapterLabeler>(
                        adapterList_
                        , rmd_->platform
                        , user_->minSoftAccuracy
                        , user_->minHardAccuracy
                        , user_->minFlankingScore
                        , user_->localAlnFlanking
                        , user_->flankLength
#ifdef DIAGNOSTICS
                        , user_->emitAdapterMetrics
#endif
                        , user_->lookForLoop
                        , user_->trimToLoop
                        , user_->stemLength
                );
        sensitiveAdapterLabeler_ =
                std::make_unique<AdapterLabeler>(
                        adapterList_
                        , rmd_->platform
                        , SensitiveAdapterLabeler::minSoftAccuracy
                        , SensitiveAdapterLabeler::minHardAccuracy
                        , SensitiveAdapterLabeler::minFlankingScore
                        , SensitiveAdapterLabeler::localAlnFlanking
                        , SensitiveAdapterLabeler::flankLength
#ifdef DIAGNOSTICS
                        , user_->emitAdapterMetrics
#endif
                        , false // We should already know the stem length by now
                        , false // We should already know the stem length by now
                        , user_->stemLength
                );
        rcAdapterLabeler_ =
                std::make_unique<AdapterLabeler>(
                        rcAdapterList_
                        , rmd_->platform
                        , SensitiveAdapterLabeler::minSoftAccuracy
                        , SensitiveAdapterLabeler::minHardAccuracy
                        , SensitiveAdapterLabeler::minFlankingScore
                        , SensitiveAdapterLabeler::localAlnFlanking
                        , SensitiveAdapterLabeler::flankLength
#ifdef DIAGNOSTICS
                        , user_->emitAdapterMetrics
#endif
                        , false // We should already know the stem length by now
                        , false // We should already know the stem length by now
                        , user_->stemLength
                );
        palindromeScorer_ = std::make_unique<ScoringScheme>(
                rmd_->platform, ScoringScope::ADAPTER_FINDING);
    }
}

void SubreadLabeler::InitControlFiltering()
{
    if (performControlFiltering_)
    {
        controlFilter_ = std::unique_ptr<ControlFilter>(
                new ControlFilter(controlList_, rmd_->platform, controlAdapterList_,
                                  user_->useSplitControlWorkflow,
                                  user_->splitReadLength,
                                  user_->splitReferenceLength));
    }
}

}}} // ::PacBio::Primary::Postprimary
