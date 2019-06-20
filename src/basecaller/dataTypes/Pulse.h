// Copyright (c) 2019 Pacific Biosciences of California, Inc.
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
/// \file	Pulse.h
/// \brief	The class Pulse is used to represent pulses identified and
///         quantified in trace data.

#ifndef PacBio_Mongo_Pulse_H_
#define PacBio_Mongo_Pulse_H_


#include <cassert>
#include <cmath>
#include <limits>
#include <stdint.h>

#include <common/cuda/CudaFunctionDecorators.h>

namespace PacBio {
namespace Mongo {
namespace Data {

/// \brief A trivial type to represent trace pulses.
class Pulse
{
public: // Types

    enum class NucleotideLabel : uint8_t
    {
        A = 0, C, G, T,
        N,      // A, C, G, or T
        NONE
    };

public:     // Static constants
    /// The maximum value allowed by the internal representation of MeanSignal
    /// and MidSignal.
    CUDA_ENABLED static constexpr float SignalMax()
    {
        return FromFixedPrecision(std::numeric_limits<uint16_t>::max());
    }

    /// The precision of the internal representation of MeanSignal and
    /// MidSignal. The smallest non-zero signal level that can be represented.
    CUDA_ENABLED static constexpr float SignalPrecision()
    {
        return repScaleSignal_;
    }


public:     // Equality
    CUDA_ENABLED bool operator==(const Pulse& that) const
    {
        if (this->start_ != that.start_) return false;
        if (this->width_ != that.width_) return false;
        if (this->meanSignal_ != that.meanSignal_) return false;
        if (this->midSignal_ != that.midSignal_) return false;
        if (this->maxSignal_ != that.maxSignal_) return false;
        if (this->signalM2_ != that.signalM2_) return false;
        if (this->label_ != that.label_) return false;
        return true;
    }

    CUDA_ENABLED bool operator!=(const Pulse& that) const
    { return !(*this == that); }

public:     // Property accessors
    /// The first frame of this pulse.
    CUDA_ENABLED uint32_t Start() const
    { return start_; }

    /// The first frame after the last frame of this pulse.
    CUDA_ENABLED uint32_t Stop() const
    { return start_ + width_; }

    /// The number of frames in this pulse.
    CUDA_ENABLED uint16_t Width() const
    { return width_; }

    /// The mean signal level in the pulse.
    CUDA_ENABLED float MeanSignal() const
    { return FromFixedPrecision(meanSignal_); }

    /// \brief The mean of the signal over the interior frames of the pulse.
    /// \details If Width() < 3, MidSignal() is NaN.
    CUDA_ENABLED float MidSignal() const
    {
        if (width_ < 3) return std::numeric_limits<float>::quiet_NaN();
        else return FromFixedPrecision(midSignal_);
    }

    /// The max signal level in the pulse.
    CUDA_ENABLED float MaxSignal() const
    { return FromFixedPrecision(maxSignal_); }

    /// The sum of squares of the signal level for interior frames of the
    /// pulse.
    CUDA_ENABLED float SignalM2() const
    {
        if (width_ < 3) return std::numeric_limits<float>::quiet_NaN();
        else return signalM2_;
    }

    /// \brief The label assigned to the pulse by the pulse classifier.
    /// \details This label is guaranteed to be a DNA base; we do not lable
    ///          pulses in any other manner (no "no-call" pulses).
    CUDA_ENABLED NucleotideLabel Label() const
    { return label_; }

public:     // Property modifiers
    /// Sets the start frame value.
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& Start(uint32_t value)
    {
        start_ = value;
        return *this;
    }

    /// Sets the width in frames.
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& Width(uint16_t value)
    {
        width_ = value;
        return *this;
    }

    /// Sets the mean signal value.
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& MeanSignal(float value)
    {
        meanSignal_ = ToFixedPrecision(value);
        return *this;
    }

    /// \brief Sets the mid-pulse signal value.
    /// \detail The value will be stored, but will be hidden as long as
    /// Width() < 3.
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& MidSignal(float value)
    {
        midSignal_ = ToFixedPrecision(value);
        return *this;
    }

    /// Sets the max signal value.
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& MaxSignal(float value)
    {
        maxSignal_ = ToFixedPrecision(value);
        return *this;
    }

    /// Sets the sum of signal values
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& SignalM2(float value)
    {
        signalM2_ = value;
        return *this;
    }

    /// \brief Sets the label.
    /// \detail \code
    /// \returns Reference to \code *this.
    CUDA_ENABLED Pulse& Label(NucleotideLabel value)
    {
        label_ = value;
        return *this;
    }

private:    // Static data
    // Scale factor used for fixed-point representation of signal levels.
    // The representation has this precision in DN or e-.
    CUDA_ENABLED static constexpr float repScaleSignal_ = .1f;

    CUDA_ENABLED static constexpr uint16_t ToFixedPrecision(float val)
    {
        assert(val >= 0);
        assert(val <= SignalMax());
        assert(std::isfinite(val));
        return static_cast<uint16_t>(val / repScaleSignal_ + 0.5f);
    }
    CUDA_ENABLED static constexpr float FromFixedPrecision(uint16_t val)
    {
        return val * repScaleSignal_;
    }

private:    // Data
    uint32_t start_;       // frames
    uint16_t width_;       // frames

    uint16_t meanSignal_;  // scaled DN or e-.
    uint16_t midSignal_;   // scaled DN or e-.
    uint16_t maxSignal_;
    // TODO make PBHalf?  This class currently has 3 bits of padding
    // and this could drop it to 1.
    float signalM2_;

    NucleotideLabel label_;
};

}}}  // PacBio::Mongo::Data

#endif // PacBio_Mongo_Pulse_H_
