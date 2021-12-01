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

#ifndef PACBIO_CUDA_SUBFRAME_SCORER_H
#define PACBIO_CUDA_SUBFRAME_SCORER_H

#include <algorithm>

#include <pacbio/auxdata/AnalogMode.h>

#include <common/cuda/PBCudaSimd.h>
#include <common/cuda/utility/CudaArray.h>
#include <common/MongoConstants.h>

#include <dataTypes/LaneDetectionModel.h>

#include <basecaller/traceAnalysis/SubframeLabelManager.h>

#include "SparseMatrix.cuh"

namespace PacBio {

// I normally wouldn't do `using namespace` in a header, but these
// prototypes pre-exist some of the new Mongo infrastructure that has
// come up, and there has not yet been time to clean up and "productionize"
// them.  Eventually everything real should be moved out of the prototypes
// anyway
using namespace PacBio::Mongo;

namespace Cuda {

static constexpr unsigned int ViterbiStitchLookback = 16u;

namespace Subframe {

//TODO refactor viterbi to be less strongly coupled with a particular subframe
//     implementation
using Mongo::Basecaller::SubframeLabelManager;
static constexpr int numStates = SubframeLabelManager::numStates;

template <typename T>
using SparseTransitionSpec = SparseMatrixSpec<T,
//                B  T  G  C  A  TU GU CU AU TD GD CD AD
    SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // Baseline
    SparseRowSpec<0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>, // T
    SparseRowSpec<0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>, // G
    SparseRowSpec<0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0>, // C
    SparseRowSpec<0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0>, // A
    SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // T Up
    SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // G Up
    SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // C Up
    SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // A Up
    SparseRowSpec<0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>, // T Down
    SparseRowSpec<0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>, // G Down
    SparseRowSpec<0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0>, // C Down
    SparseRowSpec<0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0>  // A Down
>;

template <typename T>
struct __align__(128) TransitionMatrix : public SparseTransitionSpec<T>
{
    // Default constructor for use in device __constant__ memory.  Will
    // remain uninitialized until the host coppies up the data
    TransitionMatrix() = default;

    // Ctor for host construction
    TransitionMatrix(Utility::CudaArray<PacBio::AuxData::AnalogMode, numAnalogs> analogs,
                     double frameRate);
};

// Gaussian fallback for edge frame scoring.  This is a near unchanged transcription
// of what exists in Sequel.  All operations are designed to be done by cuda blocks
// with one thread per pair of zmw.
template <typename VF>
class __align__(128) NormalLog
{
    static constexpr float log2pi_f = 1.8378770664f;
public:     // Structors
    NormalLog() = default;

public:     // Properties
    CUDA_ENABLED const VF& Mean() const
    { return mean_; }

    /// Sets Mean to new value.
    /// \returns *this.
    template <typename VF2>
    CUDA_ENABLED NormalLog& Mean(const VF2& value)
    {
        mean_ = value;
        return *this;
    }

    CUDA_ENABLED const VF& Variance() const
    { return var_; }

    /// Sets Variance to new value, which must be positive.
    /// \returns *this.
    template <typename VF2>
    CUDA_ENABLED NormalLog& Variance(const VF2& value)
    {
        //assert(all(value > 0.0f));
        var_ = value;
        return *this;
    }

public:
    /// Evaluate the distribution at \a x.
    template <typename VF2>
    CUDA_ENABLED auto operator()(const VF2& x) const
    {
        auto normTerm = -0.5f * (log2pi_f + log(var_));
        auto scaleFactor = 0.5f / var_;
        return normTerm - pow2(x - mean_) * scaleFactor;
    }

private:    // Data
    VF mean_;
    VF var_;
};

template <typename VF>
constexpr float NormalLog<VF>::log2pi_f;

// Subframe scorer using gause caps. This is a near unchanged transcription
// of what exists in Sequel, though some precomputed terms have been removed
// as storage is more expensive than computation. All operations are designed
// to be done by cuda blocks with one thread per pair of zmw.
template <typename VF>
struct __align__(128) BlockStateSubframeScorer
{
    template <typename T, size_t Len>
    using LaneModelParameters = Mongo::Data::LaneModelParameters<T, Len>;

    // Default constructor only exists to facilitate a shared memory instance.
    // The object is not in a valid state until after calling `Setup`
    BlockStateSubframeScorer() = default;

    template <typename T, size_t Len>
    CUDA_ENABLED void Setup(const LaneModelParameters<T, Len>& model)
    {
        static_assert(Len > 1, "Scalar types not expected");

        static constexpr float log2pi_f = 1.8378770664f;
        const float nhalfVal = -0.5f;

        SubframeSetup(model);

        const float normConst = log2pi_f / 2.0f;

        // Background
        const auto& bgMode = model.BaselineMode();

        // Put -0.5 * log(det(V)) into bgFixedTerm_; compute bgInvCov_.
        // sym2x2MatrixInverse returns the determinant and writes the inverse
        // into the second argument.
        bgFixedTerm_ = nhalfVal * log(VF::FromArray(bgMode.vars)) - normConst;
        bgMean_ = bgMode.means;

        #pragma unroll 1
        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            // Full-frame states
            const auto& aMode = model.AnalogMode(i);
            ffFixedTerm_[i] = nhalfVal * log(VF::FromArray(aMode.vars)) - normConst;
            ffmean_[i] = aMode.means;
        }
    }

    template <typename VF2>
    CUDA_ENABLED auto StateScores(const VF2& data, int state) const
    {
        const float nhalfVal = -0.5f;

        switch (state)
        {
        case 0:
            {

                const auto y = data - bgMean_;
                return nhalfVal * y * bInvVar_ * y + bgFixedTerm_;
            }
        case 1:
        case 2:
        case 3:
        case 4:
            {
                const auto i = state-1;
                const auto y = data - ffmean_[i];
                return nhalfVal * y * pInvVar_[i]*y + ffFixedTerm_[i];
            }
        default:
            {
                using std::min;
                const auto i = (state-1)%4;

                auto score = SubframeScore(i, data);

                // When signal is stronger than full-frame mean, ensure that
                // subframe density does not exceed full-frame density.
                const auto xmu = data * ffmean_[i];
                const auto mumu = ffmean_[i]*ffmean_[i];

                const auto y = data - ffmean_[i];
                const auto fScore =  nhalfVal * y * pInvVar_[i]*y + ffFixedTerm_[i];

                // xmu > mumu means that the Euclidean projection of x onto mu is
                // greater than the norm of mu.
                score = Blend(xmu > mumu,
                              min(min(score, fScore - 1.0f), 0.0f),
                              score);

                return score;
            }
        }
    }

 private:
    template <typename T, size_t Len>
    CUDA_ENABLED void SubframeSetup(const LaneModelParameters<T, Len>& model)
    {
        static constexpr float pi_f = 3.1415926536f;
        const float sqrtHalfPi = std::sqrt(0.5f * pi_f);
        const float halfVal = 0.5f;
        const float oneSixth = 1.0f / 6.0f;

        const auto& bMode = model.BaselineMode();
        const auto bMean = VF::FromArray(bMode.means);
        const auto bVar = VF::FromArray(bMode.vars);
        const auto bSigma = sqrt(bVar);

        // Low joint in the score function.
        x0_ = bMean + bSigma * sqrtHalfPi;
        bInvVar_ = 1.0f / bVar;

        #pragma unroll 1
        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto& dma = model.AnalogMode(a);
            const auto aMean = VF::FromArray(dma.means);
            logPMean_[a] = log(aMean - bMean);
            const auto aVar = VF::FromArray(dma.vars);
            const auto aSigma = sqrt(aVar);
            // High joint in the score function.
            x1_[a] = aMean - aSigma * sqrtHalfPi;
            pInvVar_[a] = 1.0f / aVar;
        }

        // Assume that the dimmest analog is the only one at risk for
        // violating x0_ <= x1_[a].
        {
            const auto& dm = model.AnalogMode(3);
            const auto aMean = VF::FromArray(dm.means);
            const auto aVar = VF::FromArray(dm.vars);
            // Edge-frame mean and variance.
            auto eMean = halfVal * (bMean + aMean);
            auto eVar = oneSixth * (aMean - bMean);
            eVar += bVar + aVar;
            eVar *= halfVal;
            // Normal approximation fallback when x1 < x0.
            dimFallback_.Mean(eMean).Variance(eVar);
        }
    }

    template <typename VF2>
    CUDA_ENABLED auto SubframeScore(unsigned int a, const VF2& val) const
    {
        const float nhalfVal = -0.5f;

        // Score between x0 and x1.
        const auto lowExtrema = val < x0_;
        const auto cap = Blend(lowExtrema, x0_, x1_[a]);
        auto fixedTerm = Blend(lowExtrema, bInvVar_, pInvVar_[a]);
        fixedTerm = nhalfVal * fixedTerm;

        const auto extrema = lowExtrema | (val > x1_[a]);
        auto s = Blend(extrema, fixedTerm * pow2(val - cap), 0.0f) - logPMean_[a];

        if (a == numAnalogs-1)
        {
            // Fall back to normal approximation when x1 < x0.
            s = Blend(x0_ <= x1_[a], s, dimFallback_(val));
        }
        return s;
    }

    VF bgMean_;
    Utility::CudaArray<VF, numAnalogs> ffmean_; // full-frame states

    VF bInvVar_;
    Utility::CudaArray<VF, numAnalogs> pInvVar_;

    // -0.5 * log det(V) - log 2\pi.
    VF bgFixedTerm_;    // background (a.k.a. baseline)
    Utility::CudaArray<VF, numAnalogs> ffFixedTerm_;    // full-frame states

    // Log of displacement of pulse means from baseline mean.
    Utility::CudaArray<VF, numAnalogs> logPMean_;

    // Joints in the subframe score function.
    VF x0_;
    Utility::CudaArray<VF, numAnalogs> x1_;

    // Normal approximation fallback for dimmest analog subframe when x1 < x0.
    NormalLog<VF> dimFallback_;
};

}}}

#endif //PACBIO_CUDA_SUBFRAME_SCORER_H
