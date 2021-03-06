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

#include <common/AlignedVector.h>
#include <common/IntInterval.h>
#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/configs/BasecallerDmeConfig.h>

namespace PacBio::Mongo::Data {

// Defined below.
template <typename VF>
class SignalModeHost;

// TODO: Is there a better home for this type alias?
using FrameIntervalType = IntInterval<FrameIndexType>;

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

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig);

    /// The variance for \analog signal based on model including Poisson and
    /// "excess" noise.
    static FloatVec ModelSignalCovar(const VF& excessNoiseCV2,
                                     const FloatVec& signalMean,
                                     const FloatVec& baselineVar);

    static FloatVec XsnCoeffCVSq(const VF& signalMean,
                                 const VF& signalCovar,
                                 const VF& baselineCovar);

public:     // Structors and assignment
    DetectionModelHost() = delete;
    DetectionModelHost(const DetectionModelHost& other) = default;
    DetectionModelHost(DetectionModelHost&& other) = default;
    ~DetectionModelHost() = default;

    /// Frame interval needs to be specified separately because it is tracked at
    /// the pool level and is not included in LaneDetectionModel.
    template <typename FloatT>
    DetectionModelHost(const LaneDetectionModel<FloatT>& ldm,
                       const FrameIntervalType& fi);
    
    template <typename FloatT>
    DetectionModelHost(const DetectionModelPool<FloatT>& dmp, size_t i)
        : DetectionModelHost(dmp.data.GetHostView()[i], dmp.frameInterval)
    { }

    DetectionModelHost& operator=(const DetectionModelHost&) = default;
    DetectionModelHost& operator=(DetectionModelHost&&) = default;

public:     // Copy scalar slice
    /// Extract a scalar detection model for a specific unit cell (i.e., ZMW).
    DetectionModelHost<float> ZmwDetectionModel(size_t zmwIndex) const;

public:     // Operators
    /// Equality is determined by comparing baseline mode and detection modes.
    /// Confidence score and frame interval are ignored.
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

    /// The heuristic confidence score for the model for each unit cell.
    /// 0 indicates no confidence; 1, utmost confidence from a single
    /// estimatation.
    /// If the model is a result of multiple estimates averaged together, the
    /// confidence score may exceed 1.0.
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
    /// Updates *this with the other parameters
    /// Does not modify Confidence().
    /// The particular update method can be indicated in config
    /// \returns *this.
    DetectionModelHost& Update(const DetectionModelHost& other, VF fraction);

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
        // Use interval midpoints.
        const auto t0 = FrameInterval().CenterInt();
        const auto t1 = newInterval.CenterInt();
        const float tDiff = static_cast<float>(t1 - t0);
        const auto m = exp2(-abs(tDiff)/confHalfLife);
        confid_ *= m;
        SetNonemptyFrameInterval(newInterval);
    }

private:
    /// Sets the frame interval.
    /// \returns *this.
    DetectionModelHost& SetNonemptyFrameInterval(const FrameIntervalType& fi)
    {
        assert (!fi.Empty());
        frameInterval_ = fi;
        return *this;
    }

    /// Updates *this with the SIMD weighted average
    /// "\a fraction * other + (1 - \a fraction) * *this".
    /// 0 <= \a fraction <= 1.
    void Update0(const DetectionModelHost& other, VF fraction);

    // Updates *this with geometric averaging for variances and analog means.
    // Still uses arithmetic averaging for baseline mean.
    // Uses noise model to set analog variances.
    void Update1(const DetectionModelHost& other, VF fraction);

    // Yet another update method
    void Update2(const DetectionModelHost& other, VF fraction);

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

    // Frame interval associated with estimateConfid_.
    FrameIntervalType frameInterval_ {0, 1};

private:    // Static data
    static uint32_t updateMethod_;
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

    SignalModeHost(const FloatVec& weight, const FloatVec& mean, const FloatVec& variance)
        : weight_ (weight)
        , mean_ (mean)
        , var_ (variance)
    { }

    template <typename VF2>
    SignalModeHost(const LaneAnalogMode<VF2, laneSize>& lam);

    ~SignalModeHost() = default;

public:     // Vectorized comparisons
    BoolVec operator==(const SignalModeHost& other) const
    {
        // TODO: Should we compare weights?
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
        assert(all(0.0f <= w) && all(w <= 1.0f));
        weight_ = w;
    }

    /// Update the mean signal vector to a new value
    void SignalMean(const FloatVec& x)
    {
        mean_ = x;
    }

    /// Update the signal covariance matrix to a new value
    void SignalCovar(const FloatVec& v)
    {
        var_ = v;
    }

private:    // Data
    // Fraction of signal associated with this mode.
    FloatVec weight_;

    // The estimate of the signal mean.
    FloatVec mean_;

    // Estimate of the signal variance.
    FloatVec var_;
};

}   // namespace PacBio::Mongo::Data

#endif  // mongo_dataTypes_DetectionModelHost_H_
