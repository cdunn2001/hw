#ifndef Sequel_Primary_DetectionMode_H_
#define Sequel_Primary_DetectionMode_H_

// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \file   DetectionMode.h
/// \brief  The class DetectionMode defines the properties relevant to pulse
///         detection and labeling for a Lane of ZMWs or a single ZMW.

#include <cassert>
#include <algorithm>
#include <array>
//#include <pacbio/logging/Logger.h>
#include <pacbio/primary/Util.h>

#if defined(__INTEL_COMPILER)
// warning #1366: a reduction in alignment without the "packed" attribute is ignored
#pragma warning( disable : 1366 ) 
#endif

namespace PacBio {
namespace Primary {

/// Basic properties for a mode of detection, e.g. background or analog,
/// for an arbitrary number of detection channels, or cameras. Channel
/// order of the data must be specified by the user of this class.
///
template <typename VF, size_t NCam = 2>
class alignas(VF) DetectionMode
{
public:     // Static functions
    unsigned int rowColToBandDiagIndx(unsigned int row, unsigned int col)
    {
        static const unsigned int nn1 = NCam * (NCam-1);
        unsigned int r = std::max(row, col);
        const auto d = std::abs(static_cast<int>(row) - static_cast<int>(col));
        r += (nn1 - (NCam - d) * (NCam - d - 1))/2;
        assert (r < NCam*(NCam+1)/2);
        return r;
    }

public:     // Structors
    
    /// Construct the DetectionMode from raw data, given in channel order
    /// that is consistent with the internal camera trace channel order.
    ///
    DetectionMode(const std::array<VF, NCam>& signalMean,
                  const std::array<VF, NumCvr(NCam)>& signalCovar,
                  const VF& weight = VF(0.1f))
        : signalMean_(ObjectAlignmentCheck(signalMean))
        , signalCovar_(signalCovar)
        , weight_(weight)
    {
        //PBLOG_TRACE << "DetectionMode constructor.";
    }

    DetectionMode()
    {
        //PBLOG_TRACE << "DetectionMode DEFAULT constructor.";
        assert((size_t)&signalMean_ % alignof(VF) == 0);
    }

/*
    DetectionMode(const DetectionMode& that)
        : signalMean_(ObjectAlignmentCheck(that.signalMean_))
        , signalCovar_(that.signalCovar_)
        , weight_(that.weight_)
    {
        //PBLOG_TRACE << "DetectionMode COPY constructor.";
    }
*/

    // Would like to use "= default" to make this type "trivial". But a gcc bug
    // causes some methods to be omitted in explicit instantiation.
    // See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=57728 .
//    DetectionMode() = default;
    DetectionMode(const DetectionMode&) = default;

//    ~DetectionMode() = default;

public:     // Assignment
//    DetectionMode& operator=(const DetectionMode&) = default;

public:     // Read Access

    /// The mode spectrum, probability (L1) normalized.
    std::array<VF, NCam> Spectrum() const
    {
        return L1Normalized(signalMean_);
    }
    /// The amplitude of the mode, i.e. the signal L1-norm (photo e-).
    VF Amplitude() const
    {
        return L1Norm(signalMean_.begin(), signalMean_.end());
    }
    /// The mean signal vector (photo e-).
    const std::array<VF, NCam>& SignalMean() const
    {
        return signalMean_;
    }
    /// The signal covariance matrix in diagonal-band ordering (photo e-)^2.
    /// For NCam==2, this order is (0,0), (1,1), (0,1).
    const std::array<VF, NumCvr(NCam)>& SignalCovar() const
    {
        return signalCovar_;
    }

    // TODO: Should this be a member of DetectionModel instead?
    /// The weight (or mixture fraction) of this mode in the context of the
    /// detection model.
    const VF& Weight() const
    { return weight_; }


public: // Modify Access
        // Setters return void to avoid delegation methods in derived classes.

    /// Update the spectrum to a new value
    void Spectrum(const std::array<VF, NCam>& s)
    {
        // Maintain the current |signal|
        VF amplitude = L1Norm(signalMean_.begin(), signalMean_.end());
        // Set the new direction
        signalMean_ = L1Normalized<VF, NCam>(s, amplitude);
    }
    /// Update the amplitude to a new value
    void Amplitude(VF a)
    {
        signalMean_ = L1Normalized<VF, NCam>(signalMean_, a);
    }    
    /// Update the mean signal vector to a new value
    void SignalMean(const std::array<VF, NCam>& x)
    {
        signalMean_ = x;
    }
    /// Update the signal covariance matrix to a new value
    void SignalCovar(const std::array <VF, NumCvr(NCam)>& v)
    {
        signalCovar_ = v;
    }

    /// Update the mixture fraction (a.k.a. weight).
    void Weight(const VF& value)
    { weight_ = value; }

    /// Blends in another detection mode with specified mixing fraction.
    /// After calling Update, *this will be the weighted average of the precall
    /// value and other.
    /// 0 <= \a fraction <= 1.
    void Update(const DetectionMode<VF, NCam>& other, const VF fraction)
    {
        // TODO: When we have SIMD compare operations, add assert statements
        // to ensure that 0 <= fraction <= 1 (see bug 27767).

        // Don't think that any exceptions can occur here.
        // So just update in-place.

        const VF a = VF(1) - fraction;
        const VF& b = fraction;
        weight_ = a * weight_  +  b * other.weight_;
        for (unsigned int i = 0; i < signalMean_.size(); ++i)
        {
            auto& x = signalMean_[i];
            x = a * x  +  b * other.signalMean_[i];
        }
        for (unsigned int i = 0; i < signalCovar_.size(); ++i)
        {
            auto& x = signalCovar_[i];
            x = a * x  +  b * other.signalCovar_[i];
        }
    }


    /// Assign value of other to this, but with filter order permuted by perm.
    void AssignPermuteFilter(const DetectionMode<VF, NCam>& other, const std::array<int, NCam>& perm)
    {
        for (unsigned int i = 0; i < NCam; ++i)
        {
            const unsigned int k = perm[i];
            assert (k < NCam);
            signalMean_[i] = other.signalMean_[k];
            for (unsigned int j = 0; j <= i; ++j)
            {
                const unsigned l = perm[j];
                assert (l < NCam);
                const auto a = rowColToBandDiagIndx(i, j);
                const auto b = rowColToBandDiagIndx(k, l);
                signalCovar_[a] = other.signalCovar_[b];
            }
        }
        weight_ = other.weight_;
    }


protected: // Data Members
    
    // The mean signal estimate in detection coordinates (photo e-).
    std::array<VF, NCam> signalMean_;
    
    // The signal covariance matrix in band-storage order (photo e-)^2.
    // For example, for NCam=3, using 1-based indexing, the values of the
    // covariance matrix V are stored as [V11, V22, V33, V12, V23, V13].
    //
    std::array<VF, NumCvr(NCam)> signalCovar_;

    // The weight of this mode relative to all modes in the detection
    // model, e.g., the cluster weight returned by mode estimation.
    // TODO: Being that this is something that only has meaning relative to the
    // other modes in the model, it seems like it should be a member of
    // DetectionModel instead.
    VF weight_;

private:    // Diagnostic methods

    // Use to verify alignment from constructor initialization block
    const std::array<VF, NCam>& ObjectAlignmentCheck(const std::array<VF, NCam>& x) const
    {
        //PBLOG_TRACE << "New DetectionMode object:";
        assert((size_t)&signalMean_ % alignof(VF) == 0);
        return x;
    }
};

// TODO: Waiting for relocation of simd_conv_traits.h, which defines PacBio::BoolConv.
// For the time being, this operator is defined in DetectionModel.h.
//template <typename VF, size_t NCam = 2>
//PacBio::BoolConv<VF> operator==(const DetectionMode<VF, NCam>& lhs, const DetectionMode<VF, NCam>& rhs)
//{
//PacBio::BoolConv<VF> r {lhs.Weight() == rhs.Weight()};
//for (size_t i = 0; i < lhs.SignalMean().size(); ++i)
//{
//    r &= (lhs.SignalMean()[i] == rhs.SignalMean()[i]);
//}
//for (size_t i = 0; i < lhs.SignalCovar().size(); ++i)
//{
//    r &= (lhs.SignalCovar()[i] == rhs.SignalCovar()[i]);
//}
//return r;
//}

//template <typename VF, size_t NCam = 2>
//PacBio::BoolConv<VF> operator!=(const DetectionMode<VF, NCam>& lhs, const DetectionMode<VF, NCam>& rhs)
//{ return !(lhs==rhs); }


}} // PacBio::Primary

#endif // Sequel_Primary_DetectionMode_H_
