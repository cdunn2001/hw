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

// Programmer: Armin Töpfer, Brett Bowman

#pragma once

#include <assert.h>
#include <iostream>
#include <memory>

// hackish workaround!  There is a bug in seqan exposed by newer compilers where it
// does some type mangling that confuses this file once it eventually gets included.
// Include it now upfront to make sure it stays robost.
//
// TODO Upgrade seqan
#include <x86intrin.h>

#include <seqan/align.h>
#include <seqan/journaled_set/score_biaffine.h>

#include <pacbio/sparse/FindSeedsConfig.h>
#include <pacbio/sparse/FindSeeds.h>
#include <pacbio/sparse/ChainSeedsConfig.h>
#include <pacbio/sparse/ChainSeeds.h>
#include <pacbio/sparse/BandedAligner.h>

#include <bazio/FastaEntry.h>

#include <postprimary/bam/Platform.h>

namespace PacBio {
namespace Primary {
namespace Postprimary {

using namespace PacBio::SparseAlignment;

/// Struct for reporting the size and quality of a control sequence hit
struct ControlHit {
    ControlHit(size_t spanArg = 0, size_t startArg = 0, size_t endArg = 0,
               float coverageArg = 0.0f, float accuracyArg = 0.0f)
        : span(spanArg)
        , start(startArg)
        , end(endArg)
        , coverage(coverageArg)
        , accuracy(accuracyArg)
    {}

    size_t span;
    size_t start;
    size_t end;
    float coverage;
    float accuracy;
};

/// Contains methods to perform control filtering.
class ControlFilter
{
public: // structors
    // Default constructor

    ControlFilter(std::shared_ptr<std::vector<FastaEntry>> controlList,
                  const Platform platform,
                  std::shared_ptr<std::vector<FastaEntry>> controlAdapterList = nullptr,
                  bool useSplitControlWorkflow = true,
                  const size_t splitReadLength = 10000,
                  const size_t splitRefLength = 15000,
                  const size_t minChainSpan = 200,
                  const float minChainCoverage = 10.0f,
                  const size_t minQueryLength = 250,
                  const float minPercentIdentity = 69.0f,
                  const float minPercentAccuracy = 64.0f)
        : controlList_(controlList)
        , controlAdapterList_(controlAdapterList)
        , useSplitControlWorkflow_(useSplitControlWorkflow)
        , splitReadLength_(splitReadLength)
        , splitRefLength_(splitRefLength)
        , controls_(ControlReference(controlList, controlAdapterList))
        , bandedAligner_(controls_)
        , minChainSpan_(minChainSpan)
        , minChainCoverage_(minChainCoverage)
        , minQueryLength_(minQueryLength)
        , minPercentIdentity_(minPercentIdentity)
        , minPercentAccuracy_(minPercentAccuracy)
        , chainConfig_(1, 0, 5, 0, -5, -10, std::numeric_limits<int>::max())
    {
        // Initialize platform-specific variables
        Init(platform);

        // And initialize our reference indices
        InitializeIndex();
    }

    // Move constructor
    ControlFilter(ControlFilter&&) = delete;
    // Copy constructor
    ControlFilter(const ControlFilter&) = delete;
    // Move assignment operator
    ControlFilter& operator=(ControlFilter&&) = delete;
    // Copy assignment operator
    ControlFilter& operator=(const ControlFilter&) = delete;
    // Destructor
    ~ControlFilter() = default;

public:  // Class-specific types
    // Configuration data-types
    typedef FindSeedsConfig<11> TConfig;
    typedef seqan::Index<seqan::StringSet<seqan::DnaString>, typename TConfig::IndexType> TIndex;

    // Internal data-types
    typedef seqan::Seed<seqan::Simple> TSeed;
    typedef seqan::String<TSeed> TSeedChain;
    typedef std::pair<size_t, TSeedChain> TChainHit;
    typedef seqan::SeedSet<TSeed> TSeedSet;
    
    // Result data-types
    typedef seqan::Align<seqan::DnaString, seqan::ArrayGaps> TAlign;
    typedef std::pair<TAlign, seqan::AlignmentStats> TAlignPair;

public:  // self-modifying methods for initialization

    /// Initialize platform-specific settings that must differ to enable
    ///  optimal performance on both RS and Sequel data.
    ///
    /// \param  platform  Which platform/error model of PacBio data will
    ///                   this instance be used for.
    ///
    void Init(const Platform platform)
    {
        switch (platform)
        {
            case Platform::RSII:
                // Platform specific parameters.
            break;
            case Platform::SEQUEL:
                // Platform specific parameters.
            case Platform::SEQUELII:
                // Platform specific parameters.
            break;
            case Platform::MINIMAL:
            case Platform::NONE:
            default:
                throw std::runtime_error("Platform not specified!");
        }
    }

    /// Indices in SeqAn are compound objects whose parts are called
    ///  fibres.  Since not all functions require all of the fibres to perform
    ///  a task and their generation is computationally expensive, they are
    ///  initialized as-needed.  However, since we use our index as a const
    ///  reference, we need to force it's initialization here.  Failure to
    ///  initialize makes Control Filtering impossible, so we throw an error
    void InitializeIndex()
    {
        // build the index now because everything after uses it as const
        controlIndex_ = TIndex(controls_);
        if (!seqan::indexRequire(controlIndex_, seqan::QGramSADir()))
            throw std::runtime_error("Failed to create QGramSADir index!");
    }

private:  // internal static methods

    /// SeqAn uses custom numerical representations of it's sequences for
    ///  speed and compression, where as PPA uses the more straight-forward
    ///  convention of using STL strings.  Thus we need to convert our strings
    ///  to the SeqAn format before using their tools.  Failure to cast means
    ///  we can't use the sequence, so we throw an error.
    ///
    /// \param  sequence  A DNA sequence as a common string
    ///
    /// \return  DnaString  A DNA sequence in SeqAn's compressed format
    ///
    static seqan::DnaString ConvertToDnaString(const std::string& sequence)
    {
        using namespace seqan;

        try
        {   
            return static_cast<DnaString>(sequence);
        }
        catch (int)
        {
            throw std::runtime_error("Invalid character in control sequence!");
        }
    }

    /// One quick sanity-check for the quality of a chain of Kmer seeds is
    ///  it's size over the reference.  Spurious hits tend to be very small, 
    ///  while true hits tend to be larger.
    ///
    /// \param  chain  An ordered chain of Kmer hits as SeqAn seeds
    ///
    /// \return  size_t  The size of the region of the reference covered 
    ///                  by the chain
    ///
    static size_t SeedChainSpanV(const seqan::String<seqan::Seed<seqan::Simple>>& chain)
    {
        using namespace seqan;

        return endPositionV(back(chain)) - beginPositionV(front(chain));
    }

    /// The other quick sanity-check for the quality of a chain is the density
    ///  of the Kmer seeds.  True hits tend to cover the majority of their
    ///  region, while spurious hits cover only a tiny portion of it.
    ///  
    /// \param  chain  An ordered chain of Kmer hits as SeqAn seeds
    ///
    /// \return  size_t  The number of non-overlapping bases covered 
    ///                  by the chain.
    ///
    static size_t SeedChainBases(const seqan::String<seqan::Seed<seqan::Simple>>& chain)
    {
        using namespace seqan;

        size_t bases = 0;
        size_t prevEnd = 0;
        for (const auto& s : chain)
        {
            size_t start = std::max(prevEnd, beginPositionH(s));
            bases += endPositionH(s) - start;
            prevEnd = endPositionH(s);
        }

        return bases;
    }

    /// Return coordinates of alignment on query read.
    ///
    /// \param  align  A SeqAn alignment object of
    ///
    /// \return  pair<size_t,size_t>  The start and end coordinates
    ///
    ///
    static std::pair<size_t,size_t> QueryCoordinates(const TAlign& align)
    {
        using namespace seqan;

        const auto& queryRow = row(align, 0);
        size_t queryStart = toSourcePosition(queryRow, 0);
        size_t queryEnd   = toSourcePosition(queryRow, length(queryRow));

        return std::pair<size_t,size_t>(queryStart, queryEnd);
    }

    /// We require all hits that we consider significant to be at least
    ///  some minimum length.  When perform a full banded alignment,
    ///  the result may be either shorter or (more usually) longer than
    ///  that of the underlying seed chain, and so must be measured
    ///  accordingly.
    ///
    /// \param  align  A SeqAn alignment object of 
    ///
    /// \return  size_t  The length of the final alignment along the 
    ///                  length of the query sequence
    ///
    static size_t QueryLength(const TAlign& align)
    {
        const auto queryCoords = QueryCoordinates(align);
        return queryCoords.second - queryCoords.first;
    }

    /// Compute the actual accuracy using the canonical customer facing
    /// measurement as 1 - (#errors / read length)
    ///
    /// \param  ap      A TAlignPair object
    ///
    /// \return float   The accuracy of the alignment
    ///
    static float QueryAccuracy(const TAlignPair& ap)
    {
        using namespace seqan;

        const size_t queryLength = QueryLength(ap.first);
        if (queryLength > 0)
        {
            unsigned int numErrors = ap.second.numMismatches + ap.second.numDeletions + ap.second.numInsertions;
            return (1.0f - ((float)numErrors / (float)queryLength)) * 100.0f;
        }
        else
        {
            return -1;
        }
    }

public:  // non-modifying, public interface methods

    /// Given a DNA sequence as a string, return the best hit
    /// to the control.
    ///
    /// \param bases  A DNA sequence as an STL string
    ///
    /// \return THit  Return the best hit.
    ControlHit AlignToControl(const std::string& bases)
    {

        ControlHit hit = (useSplitControlWorkflow_)
                         ? SplitHit(bases)
                         : BestHit(bases);
        return hit;
    }

private: // non-modifying, private interface methods

    /// Create unrolled reference based on given control length
    ///
    /// \param controlList         list of control sequences
    /// \param controlAdapterList  list of control adapter sequences
    /// \param ctrlLength          control length to unroll
    /// \return StringSet          A Seqan object
    seqan::StringSet<seqan::DnaString>
    ControlReference(std::shared_ptr<std::vector<FastaEntry>> controlList,
                     std::shared_ptr<std::vector<FastaEntry>> controlAdapterList,
                     size_t ctrlLength=100000) const
    {
        using namespace seqan;

        // Assume single control and symmetric adapter.
        const FastaEntry& control = controlList->front();
        const FastaEntry& adapter = controlAdapterList->front();

        const std::string oneRoll = adapter.sequence +
                                    control.sequence +
                                    adapter.sequence +
                                    SequenceUtilities::ReverseCompl(control.sequence);

        size_t unrolledCtrlLength = ctrlLength;
        if (useSplitControlWorkflow_)
            unrolledCtrlLength = splitRefLength_;

        std::string curSeq;
        while (curSeq.size() < unrolledCtrlLength)
        {
            curSeq += oneRoll;
        }

        DnaString dnaString = ConvertToDnaString(curSeq);
        StringSet<DnaString> stringSet;
        appendValue(stringSet, dnaString);

        return stringSet;
    }

    ///  Given a DNA sequence as a string, returns the best hit that
    ///  exceeds our quality thresholds.
    ///  This is the core implementation behind AlignToControl to allow
    ///  for better debugging and testing.
    ///
    /// \param  bases  A DNA sequence as an STL string
    ///
    /// \return  THit  Return the best hit
    ControlHit BestHit(const std::string& bases)
    {
        using namespace seqan;

        // Get seqan DnaString
        const auto querySeq = ConvertToDnaString(bases);

        // Find alignment seeds...
        std::map<size_t, TSeedSet> seedSets;
        FindSeeds<TConfig>(&seedSets, controlIndex_, querySeq);

        // ...then chains of seeds, for the current read
        std::vector<std::pair<size_t, TSeedSet>> chains;
        ChainSeeds(&chains, seedSets, chainConfig_);

        // Initialize our return object which defaults to no hit.
        ControlHit emptyHit;

        // Default is to return only a single candidate.
        if (!chains.empty() && chains.size() == 1)
        {
            const auto& chainPair = chains.front();

            // If we have no seeds we can't have a hit.
            if (length(chainPair.second) < 1)
                return emptyHit;

            // We need to run the SeqAn chain to remove overlaps, but we can't
            //  do it in place, so create a similar pair with the chain result
            TChainHit chainHit;
            chainHit.first = chainPair.first;
            chainSeedsGlobally(chainHit.second, chainPair.second, SparseChaining());

            // If our chain is empty, we can't have a hit
            if (length(chainHit.second) < 1)
                return emptyHit;

            size_t spanV = SeedChainSpanV(chainHit.second);
            size_t b2 = SeedChainBases(chainHit.second);
            float coverage = 100.0f * b2 / static_cast<float>(spanV);

            // Noise hits average reference spans of ~100-150, and coverage of ~0.35-0.45
            // Above those numbers the probability a hit is real approaches ~1.0
            if (spanV < minChainSpan_ || coverage < minChainCoverage_)
                return emptyHit;

            // Perform the alignment and run the full banded S-W.
            TSeed region;
            const auto& pair = bandedAligner_.AlignHit(region, querySeq, chainHit);
            const auto readCoords = QueryCoordinates(pair.first);
            size_t queryLength = QueryLength(pair.first);
            float queryAccuracy = QueryAccuracy(pair);

            if (pair.second.alignmentIdentity < minPercentIdentity_ ||
                queryLength < minQueryLength_ ||
                queryAccuracy < minPercentAccuracy_)
                return emptyHit;

            // If we made it this far, record the hit and return
            return ControlHit(queryLength, readCoords.first, readCoords.second,
                              coverage, queryAccuracy);
        }

        return emptyHit;
    }

    ///  Given a DNA sequence as a string, returns the best hit that
    ///  exceeds our quality thresholds.
    ///  This is the core implementation behind AlignToControl to allow
    ///  for better debugging and testing.
    ///
    /// \param  bases  A DNA sequence as an STL string
    ///
    /// \return  THit  Return the best hit
    ControlHit SplitHit(const std::string& bases)
    {

        // Split the initial read into smaller segments and record
        // the start position in the original read.
        std::vector<std::pair<std::string,size_t>> splitReads;
        size_t numFullSegments = bases.size() / splitReadLength_;
        for (size_t r = 0; r < numFullSegments; r++)
        {
            splitReads.push_back(std::pair<std::string, size_t>(bases.substr(r * splitReadLength_, splitReadLength_),
                                                                r * splitReadLength_));
        }

        // Handle last segment.
        if (bases.size() % splitReadLength_ != 0)
        {
            const size_t minRuntSegmentLength = 2000;
            const size_t runtSegmentLength = bases.size() - (numFullSegments * splitReadLength_);
            const std::string runtString = bases.substr(numFullSegments * splitReadLength_, runtSegmentLength);
            if (runtSegmentLength < minRuntSegmentLength && splitReads.size() > 0)
            {
                // Add onto last segment.
                splitReads.back().first += runtString;
            }
            else
            {
                // Make new segment.
                splitReads.push_back(std::pair<std::string, size_t>(runtString,
                                                                    numFullSegments * splitReadLength_));
            }
        }


        std::vector<ControlHit> hits;
        std::transform(splitReads.begin(), splitReads.end(), std::back_inserter(hits),
                       [this](const std::pair<std::string,size_t>& read)
                       {
                            ControlHit hit = this->BestHit(read.first);
                            // Map the aligned coordinates back to
                            // coordinates from original read.
                            hit.start += read.second;
                            hit.end += read.second;
                            return hit;
                       });

        auto isHit = [](const ControlHit& hit) { return hit.accuracy > 0; };

        // Form the final hit from the segment read hits.
        auto firstHit = std::find_if(hits.begin(), hits.end(), isHit);
        auto lastHit = std::find_if(hits.rbegin(), hits.rend(), isHit);

        ControlHit emptyHit;
        if (firstHit == hits.end() && lastHit == hits.rend())
        {
            // No hit
            return emptyHit;
        }
        else
        {
            size_t readStartIdx = std::distance(hits.begin(), firstHit);
            size_t readEndIdx = std::distance(hits.begin(), lastHit.base()) - 1;

            // Compute the accuracy as weighted mean.
            float accuracy = 0;
            float weights = 0;
            for (size_t r = readStartIdx; r <= readEndIdx; r++)
            {
                accuracy += hits[r].accuracy * hits[r].span;
                weights += hits[r].span;
            }
            accuracy /= weights;

            // Returned aligned length based on start coordinate
            // of first hit and end coordinate of last hit.
            return ControlHit(lastHit->end - firstHit->start,
                              firstHit->start, lastHit->end,
                              0, accuracy);
        }
    }

private:  // Instance variables

    // Constructor-initialized variables
    const std::shared_ptr<std::vector<FastaEntry>> controlList_;
    const std::shared_ptr<std::vector<FastaEntry>> controlAdapterList_;
    const bool useSplitControlWorkflow_;
    const size_t splitReadLength_;
    const size_t splitRefLength_;
    const seqan::StringSet<seqan::DnaString> controls_;
    BandedAligner bandedAligner_;
    const size_t minChainSpan_;
    const float minChainCoverage_;
    const size_t minQueryLength_;
    const float minPercentIdentity_;
    const float minPercentAccuracy_;
    const ChainSeedsConfig chainConfig_;

    // Post-construction initialized variables
    TIndex controlIndex_;
};

}}} // ::PacBio::Primary::Postprimary

