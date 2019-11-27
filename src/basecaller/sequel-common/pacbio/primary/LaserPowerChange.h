#ifndef Sequel_Common_Pacbio_Primary_LaserPowerChange_H_
#define Sequel_Common_Pacbio_Primary_LaserPowerChange_H_

// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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
//  Defines a type that represents a change in laser power.

#include <array>
#include <vector>

#include <pacbio/primary/IntInterval.h>

namespace PacBio {
namespace Primary {

class EventObject; // forward declation

/// Represents a change in power of one or both lasers during a specific
/// frame interval. Unit of power quantities is milliwatt.
class LaserPowerChange
{
public:     // structors
    LaserPowerChange() = default;

    /// Constructs an object with specified properties.
    /// Power is assumed to be defined in units of milliwatts.
    /// If the power was changed for only one laser, set array elements for the
    /// other one to values < 0.
    /// If either topPower[0] or topPower[1] is < 0, neither is applied.
    /// Same for bottomPower.
    /// \tparam IntArray2 is an integer array type the provides operator[] for
    /// arguments 0 and 1.
    /// \tparam FloatArray2 is a float array type the provides operator[] for
    /// arguments 0 and 1.
    template <typename IntArray2, typename FloatArray2>
    LaserPowerChange(double timeStamp,
                     const IntArray2& frameInterval,
                     const FloatArray2& topPower_mW,
                     const FloatArray2& bottomPower_mW);

    LaserPowerChange(const EventObject& eo);

public:     // const methods
    /// Number of seconds since Unix epoch.
    double TimeStamp() const
    { return timeStamp_; }

    /// The "half-open" interval of frames during which the change occurred.
    const IntInterval<uint64_t>& FrameInterval() const
    { return frameInterval_; }

    /// The frame when the change started.
    uint64_t StartFrame() const
    { return frameInterval_.Lower(); }

    /// The frame when the change finished.
    uint64_t StopFrame() const
    { return frameInterval_.Upper(); }

    /// The relative (multiplicative) power change of the top laser.
    float RelPowerChangeTop() const
    {
        return topPower_[0] < 0.0f || topPower_[1] < 0.0f
                ? 1.0f : topPower_[1] / topPower_[0];
    }

    /// The relative (multiplicative) power change of the bottom laser.
    float RelPowerChangeBottom() const
    {
        return bottomPower_[0] < 0.0f || bottomPower_[1] < 0.0f
                ? 1.0f : bottomPower_[1] / bottomPower_[0];
    }

    std::string ToString() const;

public:
    /// Power in milliwatts of top laser (before, after) the change.
    const auto& TopPower()    const { return topPower_; }

    /// Power in milliwatts of bottom laser (before, after) the change.
    const auto& BottomPower() const { return bottomPower_; }

public:     // modifying methods
    /// Shift the frame interval down by \a a.
    LaserPowerChange& TranslateFrameInterval(uint64_t a)
    {
        frameInterval_.SubtractWithSaturation(a);
        return *this;
    }

private:    // data
    // Seconds since Unix epoch.
    double timeStamp_;

    // Frame interval during which the laser power is changing.
    IntInterval<uint64_t> frameInterval_;

    // Powers of the two lasers before [0] and after [1] change.
    // If the power of a particular laser was not changed,
    // its array == {-1.0f, -1.0f};
    // Unit of power is milliwatt.
    std::array<float, 2> topPower_  {-1.0f, -1.0f};
    std::array<float, 2> bottomPower_ {-1.0f, -1.0f};
};


// namespace scope functions

/// Is a.StartFrame() < b.StartFrame()?
inline bool
CompareStartFrameLess(const LaserPowerChange& a, const LaserPowerChange& b)
{ return a.StartFrame() < b.StartFrame(); }

/// Is a.StopFrame() < b.StopFrame()?
inline bool
CompareStopFrameLess(const LaserPowerChange& a, const LaserPowerChange& b)
{ return a.StopFrame() < b.StopFrame(); }

using SchunkLaserPowerChangeSet = std::vector<LaserPowerChange>;


// Member definitions.

template <typename IntArray2, typename FloatArray2>
LaserPowerChange::LaserPowerChange(double timeStamp,
                                   const IntArray2& frameInterval,
                                   const FloatArray2& topPower_mW,
                                   const FloatArray2& bottomPower_mW)
    : timeStamp_ (timeStamp)
    , frameInterval_ (frameInterval[0], frameInterval[1])
{
    if (topPower_mW[0] >= 0.0f && topPower_mW[1] >= 0.0f)
    {
        topPower_[0] = topPower_mW[0];
        topPower_[1] = topPower_mW[1];
    }
    if (bottomPower_mW[0] >= 0.0f && bottomPower_mW[1] >= 0.0f)
    {
        bottomPower_[0] = bottomPower_mW[0];
        bottomPower_[1] = bottomPower_mW[1];
    }
}


}}  // PacBio::Primary

#endif // Sequel_Common_Pacbio_Primary_LaserPowerChange_H_
