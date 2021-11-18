// Copyright (c) 2021, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
//  Defines class EdgeFrameClassifier.

#ifndef mongo_basecaller_traceAnalysis_EdgeFrameClassifier_H_
#define mongo_basecaller_traceAnalysis_EdgeFrameClassifier_H_

#include <cmath>
#include <utility>

#include <dataTypes/BasicTypes.h>
#include <common/CircularArray.h>
#include <common/LaneArray.h>

namespace PacBio::Mongo::Basecaller
{

/// A simple, stateful, aggressive classifier that indicates whether a frame
/// might represent the edge of a pulse.
class EdgeFrameClassifier
{
public:     // Types
    using FrameArray = LaneArray<Data::BaselinedTraceElement>;

public:     // Non-const methods

    /// Sets the state of *this to that of a newly created object.
    void Reset()
    {
        prevFrame = FrameArray{};
        baselineHistory.clear();
    }

    /// \brief The primary classifier function.
    /// \details To be called on a sequence of trace frames.  This filter holds
    /// state data and operates with a one-frame delay.  In other words, it
    /// returns the classification of the frame provided by the \e previous
    /// call. When called the first time after initialization or reset,
    /// IsEdgeFrame returns {LaneMask<>{true}, FrameArray{0}}. The first frame
    /// pushed through will always be classified as "edge".
    /// \param threshold  The value above which a frame is considered unlikely
    /// to represent baseline signal.
    /// \param frame  An array of trace values for a particular frame and
    /// for a lane of ZMWs.
    /// \returns a pair in which the first element is indicates whether the
    /// second element is likely to be an edge frame.
    std::pair<LaneMask<>, FrameArray>
    IsEdgeFrame(const FrameArray& threshold, const FrameArray& frame)
    {
        // This definition is not very attentive to avoiding copying FrameArray
        // objects.  It is hoped that the compiler will inline this function and
        // some copies will be optimized away.

        const auto isBaseline = (frame < threshold);
        LaneMask<> edge {true};
        // TODO: Can we avoid this conditional?
        if (baselineHistory.full())
        {
            // Consider prevFrame as suspected edge frame if exactly one of its
            // neighbor frames is baseline.
            edge = (isBaseline ^ baselineHistory.front());
        }

        // Prepare for next iteration.
        baselineHistory.push_back(isBaseline);
        const auto candidateFrame = prevFrame;
        prevFrame = frame;

        return {edge, candidateFrame};
    }

private:    // Data
    FrameArray prevFrame {};
    CircularArray<LaneMask<>, 2> baselineHistory {};
};

}   // namespace PacBio::Mongo::Basecaller

#endif  // mongo_basecaller_traceAnalysis_EdgeFrameClassifier_H_
