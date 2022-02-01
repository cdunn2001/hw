// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#include <pacbio/primary/HQRFParamsMCart.h>
#include <pacbio/primary/HQRFParamsNCart.h>

#include "BlockHQRegionFinder.h"
#include "ClassificationTree.h"

#include <boost/circular_buffer.hpp>

namespace PacBio {
namespace Primary {
namespace ActivityLabeler {

// This could be a class, it doesn't really need it though, so I'll leave it as
// a namespace for now. It will probably make sense when we move from
// source-specified models to json-specified models, so you only parse the
// model once.


// Contain details and a raw enum (used for indexing) to a namespace:
namespace RegionFinder
{

enum RFFeatures {
    IS_A0 = 0,
    IS_A1,
    IS_A2,
    A0_RATE_LATENCY,
    A1_RATE_LATENCY,
    A2_RATE_LATENCY,
    A0_RATE_HISTORY,
    A1_RATE_HISTORY,
    A2_RATE_HISTORY,
    NUM_FEATURES
};

struct Span
{
    size_t start;
    size_t end;
    size_t a1count;
};

}


// Model a Region Finding CART algorithm with a specified lookahead buffer (latency).
// The specific regionfinding algorithm is a placeholder for a prototype of
// what would eventually be a real algorithm in the realtime pipeline. But it
// is valuable to test some algorithm and feature ideas.
std::pair<int, int> LabelRegions(
        const std::vector<Activity>& activityLabels,
        size_t latency)
{
    // We are going to model a latency window of size N:
    const size_t history_depth = latency;

    // Use acircular buffer to track latency and a few historic blocks.
    // The difference between latency and history in the buffer is nothing, the
    // difference in plumbing is that latency blocks are held up in this stage
    // of the realtime pipeline, while we only retain a record of activity
    // labels for historic blocks (the blocks themselves have moved on to other
    // stages in the realtime pipeline).
    //
    // if history_depth = 3 and latency = 3, and we have a "currently
    // emitting" slot
    // buffered values: X   X   X   X   X   X   X
    //                  |   |   |   |   |   |   |- just added to buffer
    //                  |   |   |   |   |   |- blocked (latency)
    //                  |   |   |   |   |- blocked (latency)
    //                  |   |   |   |- blocked, now emitting
    //                  |   |   |- history, AL retained
    //                  |   |- history, AL retained
    //                  |- history, AL retained
    // Therefore circular buffer size is history_depth + latency + 1
    boost::circular_buffer<Activity> buffer(history_depth + latency + 1);

    // We'll maintain some proto-features between blocks, simply updated as needed:
    std::array<float, 3> latencyCounts;
    latencyCounts.fill(0);
    std::array<float, 3> historyCounts;
    historyCounts.fill(0);

    size_t latencyStart = 0;
    // Not used, but included for clarity:
    // size_t latencyEnd = latency - 1;
    size_t toEmit = latency;
    size_t historyStart = latency + 1;
    size_t historyEnd = latency + history_depth;

    // We keep track of a1count to be able to select the "best" candidate HQ
    // region. In reality we might prefer "numbases" or some other metric.
    size_t a1count = 0;
    std::vector<RegionFinder::Span> spans;
    // Our little PoC region finder is a two state Finite State Machine with
    // simple decision trees guarding transitions.
    bool hqrOpen = false;
    size_t spanStart = 0;

    const size_t numBlocks = activityLabels.size();
    for (size_t t = 0; t < numBlocks; t++)
    {
        // Update proto-feature counts:
        // (Note that we decrement history counts only after they start falling
        // off the buffer, and increment them only once they're history, and
        // not while they're still in the latency window!)
        if (t >= buffer.capacity())
        {
            historyCounts[buffer[historyEnd]]--;
        }

        buffer.push_front(activityLabels[t]);

        latencyCounts[buffer[latencyStart]]++;

        // We can't emit any labels (thereby releasing blocks) until we
        // overfill the latency cache
        if (t < latency)
            continue;

        if (t >= historyStart)
        {
            historyCounts[buffer[historyStart]]++;
        }
        latencyCounts[buffer[toEmit]]--;

        std::array<float, RegionFinder::NUM_FEATURES> features;
        features.fill(0);

        // Set the IS_A* feature:
        features[buffer[latency]] = 1;
        features[RegionFinder::A0_RATE_LATENCY] = latencyCounts[0] / latency;
        features[RegionFinder::A1_RATE_LATENCY] = latencyCounts[1] / latency;
        features[RegionFinder::A2_RATE_LATENCY] = latencyCounts[2] / latency;

        size_t hist_size = std::min(history_depth, t - (historyStart - 1));
        if (hist_size > 0.0f)
        {
            features[RegionFinder::A0_RATE_HISTORY] = historyCounts[0] / hist_size;
            features[RegionFinder::A1_RATE_HISTORY] = historyCounts[1] / hist_size;
            features[RegionFinder::A2_RATE_HISTORY] = historyCounts[2] / hist_size;
        }

        // Here are our simple decision trees dictating transitions to other
        // states. Really just a PoC placeholder for a real model using the
        // above features.
        size_t decisionIndex = t - latency;
        if (hqrOpen)
        {
            if (features[RegionFinder::IS_A1])
            {
                ++a1count;
            }
            if (features[RegionFinder::A2_RATE_LATENCY] >= 0.4
                    && (features[RegionFinder::A2_RATE_LATENCY]
                        + features[RegionFinder::A2_RATE_HISTORY]) >= 0.9
                    && features[RegionFinder::IS_A2])
            {
                spans.push_back(RegionFinder::Span({spanStart, decisionIndex, a1count}));
                hqrOpen = false;
            }
        }
        else if (!hqrOpen)
        {
            if (features[RegionFinder::A1_RATE_LATENCY] >= 0.4
                    && (features[RegionFinder::A2_RATE_LATENCY]
                        + features[RegionFinder::A2_RATE_HISTORY]) == 0.0
                    && features[RegionFinder::IS_A1])
            {
                hqrOpen = true;
                spanStart = decisionIndex;
                a1count = 1;
            }
        }
    }
    // At this point we haven't judged the last <latency> blocks in the zmw.
    // Because we've run out of labels to buffer, the results really won't
    // change much. We'll continue with whatever state we're in (hqr
    // open/closed)
    if (hqrOpen)
    {
        spans.push_back(RegionFinder::Span({spanStart, numBlocks, a1count}));
    }

    // Pick the longest candidate:
    std::pair<int, int> span(0, 0);
    size_t maxSize = 0;
    for (const auto& candidate : spans)
    {
        if (candidate.a1count > maxSize)
        {
            maxSize = candidate.a1count;
            span = std::make_pair<int, int>(candidate.start, candidate.end);
        }
    }
    return span;
}


}}} // ::PacBio::Primary::ActivityLabeler
