#ifndef mongo_dataTypes_DetectionModelHost_H_
#define mongo_dataTypes_DetectionModelHost_H_

// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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
//  Defines classes DetectionModelHost, SignalModeHost, and related types.

#include <array>
#include <common/AlignedVector.h>
#include <common/simd/SimdConvTraits.h>
#include <dataTypes/LaneDetectionModel.h>

namespace PacBio {
namespace Mongo {
namespace Data {

// Defined below.
template <typename VF>
class SignalModeHost;

/// The integer type used to refer to a specific data frame.
using FrameIndexType = uint32_t;

/// The array type used to refer to an interval of data frames.
/// The first element is the index of the first frame in the interval.
/// The second element is the index of one past the last frame in the interval.
using FrameIntervalType = std::array<FrameIndexType, 2>;

/// The integer type used to represent the number of data frames in an interval.
using FrameIntervalSizeType = uint32_t;


///  A type that represents the detection model parameters with an interface
/// that is friendly to the host implementation for detection model estimation.
/// \tparam VF A floating-point type or a LaneArray of floating-point type.
template <typename VF>
class DetectionModelHost
{
public:     // Types
    using FloatVec = VF;
    using BoolVec = Simd::BoolConv<VF>;
    using IntVec = Simd::IndexConv<VF>;

public:     // Static constants
    static constexpr unsigned int vecSize = Simd::SimdTypeTraits<VF>::width;


public:     // Structors and assignment
    DetectionModelHost() = delete;
    DetectionModelHost(const DetectionModelHost& other) = default;
    DetectionModelHost(DetectionModelHost&& other) = default;
    ~DetectionModelHost() = default;

    template <typename FloatT>
    DetectionModelHost(const LaneDetectionModel<FloatT>& ldm);

    DetectionModelHost& operator=(const DetectionModelHost&) = default;
    DetectionModelHost& operator=(DetectionModelHost&&) = default;

public:     // Copy scalar slice
    /// Extract a scalar detection model for a specific unit cell (i.e., ZMW).
    DetectionModelHost<float> ZmwDetectionModel(size_t zmwIndex) const;

public:     // Operators
    /// Equality is determined by comparing baseline mode and detection modes.
    /// Other metadata (e.g., Updated()) are ignored.
    BoolVec operator==(const DetectionModelHost& rhs) const
    {
        if (detectionModes_.size() != rhs.detectionModes_.size())
        {
            return BoolVec{false};
        }

        BoolVec r {BaselineMode() == rhs.BaselineMode()};
        for (unsigned int i = 0; i < detectionModes_.size(); ++i)
        {
            r &= (detectionModes_[i] == rhs.detectionModes_[i]);
        }

        // Do not compare other metadata (the stuff modified by ClearMetadata()).

        // Do not compare confidence and frame interval.

        return r;
    }

    BoolVec operator!=(const DetectionModelHost& rhs) const
    { return !(*this == rhs); }

public:     // Const interface
    /// The model initialization state
    bool IsInitialized() const
    { return detectionModes_.size() > 0; }

    /// Const reference to the baseline mode of the model
    const SignalModeHost<FloatVec>& BaselineMode() const
    { return baselineMode_; }

    /// The detection modes that comprise the model.
    const AlignedVector<SignalModeHost<FloatVec>>& DetectionModes() const
    {
        assert(IsInitialized());
        return detectionModes_;
    }

    /// Indicates whether the model was updated since the previous trace block
    /// for unit cell specified by \a zmwIndex.
    /// 0 <= \a zmwIndex < vecSize.
    bool Updated(size_t zmwIndex) const
    {
        assert (zmwIndex < vecSize);
        return updated_[zmwIndex];
    }

    /// Vector of booleans indicated whether each unit cell was updated.
    BoolVec Updated() const
    { return updated_; }

    /// The heuristic confidence score for the model for unit cell specificed
    /// by \a zmwIndex.
    /// 0 indicates no confidence; 1, utmost confidence from a single
    /// estimatation.
    /// If the model is a result of multiple estimates averaged together, the
    /// confidence score may exceed 1.0.
    /// 0 <= \a zmwIndex < vecSize.
    float Confidence(size_t zmwIndex) const
    {
        assert (zmwIndex < vecSize);
        return confid_[zmwIndex];
    }

    /// The confidence scores for the complete lane.
    const FloatVec& Confidence() const
    { return confid_; }

    /// The frame interval associated with this model and its Confidence().
    const FrameIntervalType& FrameInterval() const
    { return frameInterval_; }

    /// Transcribe data to \a *ldm.
    /// ldm != nullptr.
    template <typename VF2>
    void ExportTo(LaneDetectionModel<VF2>* ldm) const;

public:     // Non-const interface
    /// Updates *this with the SIMD weighted average
    /// "\a fraction * other + (1 - \a fraction) * *this".
    /// Does not modify Confidence().
    /// 0 <= \a fraction <= 1.
    /// \returns *this.
    DetectionModelHost& Update(
        const DetectionModelHost& other,
        VF fraction);

    /// Similar to the other overload of Update. This version defines the
    /// averaging weights in proportion to the Confidence of *this
    /// and other. Then it adds the confidence of other into Confidence().
    /// If both confidences are zero, *this is not modified.
    /// The frame intervals of this and other must match.
    /// \returns *this.
    DetectionModelHost& Update(const DetectionModelHost& other);

    /// Sets the estimation confidence score for all unit cells.
    /// value >= 0.
    /// \returns *this.
    DetectionModelHost& Confidence(VF value)
    {
        assert (all(value >= 0.0f));
        confid_ = value;
        return *this;
    }

    /// Sets the frame interval.
    /// fi[0] < f[1].
    /// \returns *this.
    DetectionModelHost& FrameInterval(const FrameIntervalType& fi)
    {
        assert (fi[0] < fi[1]);
        frameInterval_ = fi;
        return *this;
    }

    /// Non-const reference to the baseline mode of the model
    SignalModeHost<FloatVec>& BaselineMode()
    { return baselineMode_; }

    /// Non-const reference to the vector of analogs that comprise the model
    AlignedVector<SignalModeHost<FloatVec>>& DetectionModes()
    { return detectionModes_; }

    /// Update frame interval and decay confidence.
    void EvolveConfidence(const FrameIntervalType newInterval,
                          const FloatVec& confHalfLife)
    {
        using std::abs;
        using std::exp2;
        const auto& fi0 = FrameInterval();
        // Use interval midpoints.
        const float t0 = 0.5f * (fi0[0] + fi0[1]);
        const float t1 = 0.5f * (newInterval[0] + newInterval[1]);
        const auto m = exp2(-abs(t1-t0)/confHalfLife);
        confid_ *= m;
        FrameInterval(newInterval);
    }

private:    // Members

    // The baseline mode refers to the background mode _after_ the baseline
    // subtraction in upstream analysis.  It is initialized to zero mean.
    // If a computed residual is subtracted from camera trace data, the mean
    // of this mode should be set back to zero, and the delta accounted for
    // in the backgroundMean member of the CameraTraceBlock.
    //
    SignalModeHost<FloatVec> baselineMode_;

    // Detection modes are uninitialized on construction
    AlignedVector<SignalModeHost<FloatVec>> detectionModes_;

    // Heuristic indicator of confidence in model.
    FloatVec confid_ {0.0f};

    // Indicates whether the model for each unit cell has been updated by the
    // most recent estimation.
    BoolVec updated_;

    // Frame interval associated with estimateConfid_.
    FrameIntervalType frameInterval_ {0, 1};
};


/// Basic properties for a mode of a detection signal (background or analog).
template <typename VF>
class SignalModeHost
{
public:     // Types
    using FloatVec = VF;
    using BoolVec = Simd::BoolConv<VF>;

public:     // Structors
    SignalModeHost() = default;

    SignalModeHost(const SignalModeHost&) = default;

    SignalModeHost(const FloatVec& mean, const FloatVec& variance)
        : mean_ (mean)
        , var_ (variance)
    { }

    template <typename VF2>
    SignalModeHost(const LaneAnalogMode<VF2, laneSize>& lam);

    ~SignalModeHost() = default;

public:     // Vectorized comparisons
    BoolVec operator==(const SignalModeHost& other) const
    {
        BoolVec r = (mean_ == other.mean_);
        r &= (var_ == other.var_);
        return r;
    }

public:     // Read Access
    const FloatVec& Weight() const
    { return weight_; }

    /// The mean signal.
    const FloatVec& SignalMean() const
    { return mean_; }

    /// The signal variance signal.
    const FloatVec& SignalCovar() const
    { return var_; }

    /// Transcribe data to *lam.
    /// lam != nullptr.
    template <typename VF2>
    void ExportTo(LaneAnalogMode<VF2, laneSize>* lam) const;

public: // Modify Access
    void Weight(const FloatVec& w)
    {
        assert(all(w >= 0.0f));
        weight_ = w;
    }

    /// Update the mean signal vector to a new value
    void SignalMean(const FloatVec& x)
    { mean_ = x; }

    /// Update the signal covariance matrix to a new value
    void SignalCovar(const FloatVec& v)
    {
        assert(all(v >= 0.0f));
        var_ = v;
    }

    /// Blends in another detection mode with specified mixing fraction.
    /// After calling Update, *this will be the weighted average of the precall
    /// value and other.
    /// 0 <= \a fraction <= 1.
    void Update(const SignalModeHost<FloatVec>& other, const FloatVec fraction)
    {
        // TODO: When we have SIMD compare operations, add assert statements
        // to ensure that 0 <= fraction <= 1 (see bug 27767).

        // Don't think that any exceptions can occur here.
        // So just update in-place.

        const FloatVec a = FloatVec(1) - fraction;
        const FloatVec& b = fraction;
        mean_ = a * mean_  +  b * other.mean_;
        var_ = a * var_  +  b * other.var_;
    }

private:    // Data
    // Fraction of signal associated with this mode.
    FloatVec weight_;

    // The estimate of the signal mean.
    FloatVec mean_;

    // Estimate of the signal variance.
    FloatVec var_;
};

}}}     // namespace PacBio::Mongo::Data

#endif  // mongo_dataTypes_DetectionModelHost_H_
