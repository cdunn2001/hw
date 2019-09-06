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

#include <common/cuda/PBCudaSimd.cuh>
#include <common/cuda/utility/CudaArray.h>
#include <common/MongoConstants.h>

#include <dataTypes/LaneDetectionModel.h>

#include <basecaller/traceAnalysis/SubframeLabelManager.h>

#include "AnalogMeta.h"

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

struct __align__(128) TransitionMatrix : public Transition_t
{
    // Default constructor for use in device __constant__ memory.  Will
    // remain uninitialized until the host coppies up the data
    TransitionMatrix() = default;

    // Ctor for host construction
    TransitionMatrix(Utility::CudaArray<AnalogMeta, numAnalogs> meta);
};

// Gaussian fallback for edge frame scoring.  This is a near unchanged transcription
// of what exists in Sequel.  All operations are designed to be done by cuda blocks
// with one thread per pair of zmw.
template <size_t gpuLaneWidth>
class __align__(128) NormalLog
{
    using Row = Utility::CudaArray<PBHalf2, gpuLaneWidth>;
    static constexpr float log2pi_f = 1.8378770664f;
public:     // Structors
    NormalLog() = default;
    __device__ NormalLog(const PBHalf2& mean,
                         const PBHalf2& variance)
    {
        Mean(mean);
        Variance(variance);
    }

public:     // Properties
    __device__ const PBHalf2& Mean() const
    { return mean_[threadIdx.x]; }

    /// Sets Mean to new value.
    /// \returns *this.
    __device__ NormalLog& Mean(const PBHalf2& value)
    {
        mean_[threadIdx.x] = value;
        return *this;
    }

    __device__ PBHalf2 Variance() const
    { return var_[threadIdx.x]; }

    /// Sets Variance to new value, which must be positive.
    /// \returns *this.
    __device__ NormalLog& Variance(const PBHalf2& value)
    {
        //assert(all(value > 0.0f));
        var_[threadIdx.x] = value;
        return *this;
    }

public:
    /// Evaluate the distribution at \a x.
    __device__ PBHalf2 operator()(const PBHalf2& x) const
    {
        auto normTerm = PBHalf2(-0.5f) * (PBHalf2(log2pi_f) + log(var_[threadIdx.x]));
        auto scaleFactor = PBHalf2(0.5f) / var_[threadIdx.x];
        return normTerm - pow2(x - mean_[threadIdx.x]) * scaleFactor;
    }

private:    // Data
    Row mean_;
    Row var_;
};

// Subframe scorer using gause caps. This is a near unchanged transcription
// of what exists in Sequel, though some precomputed terms have been removed
// as storage is more expensive than computation. All operations are designed
// to be done by cuda blocks with one thread per pair of zmw.
template <size_t gpuLaneWidth>
struct __align__(128) BlockStateSubframeScorer
{
    using Row = Utility::CudaArray<PBHalf2, gpuLaneWidth>;
    using LaneModelParameters = Mongo::Data::LaneModelParameters<PBHalf2, gpuLaneWidth>;

    // Default constructor only exists to facilitate a shared memory instance.
    // The object is not in a valid state until after calling `Setup`
    BlockStateSubframeScorer() = default;

    __device__ void Setup(const LaneModelParameters& model)
    {
        static constexpr float log2pi_f = 1.8378770664f;
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 one = PBHalf2(1.0f);

        SubframeSetup(model);

        const PBHalf2 normConst = PBHalf2(log2pi_f / 2.0f);

        // Background
        const auto& bgMode = model.BaselineMode();

        // Put -0.5 * log(det(V)) into bgFixedTerm_; compute bgInvCov_.
        // sym2x2MatrixInverse returns the determinant and writes the inverse
        // into the second argument.
        bgFixedTerm_[threadIdx.x] = nhalfVal * log(bgMode.vars[threadIdx.x]) - normConst;
        bgMean_[threadIdx.x] = bgMode.means[threadIdx.x];

        #pragma unroll 1
        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            // Full-frame states
            const auto& aMode = model.AnalogMode(i);
            ffFixedTerm_[i][threadIdx.x] = nhalfVal * log(aMode.vars[threadIdx.x]) - normConst;
            ffmean_[i][threadIdx.x] = aMode.means[threadIdx.x];
        }
    }

    __device__ PBHalf2 StateScores(PBHalf2 data, int state) const
    {
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);

        switch (state)
        {
        case 0:
            {

                const auto mu = bgMean_[threadIdx.x];
                const auto y = data - mu;
                return nhalfVal * y * bInvVar_[threadIdx.x] * y + bgFixedTerm_[threadIdx.x];
            }
        case 1:
        case 2:
        case 3:
        case 4:
            {
                const auto i = state-1;
                const auto j = SubframeLabelManager::FullState(i);
                const auto mu = ffmean_[i][threadIdx.x];
                const auto y = data - mu;
                return nhalfVal * y * pInvVar_[i][threadIdx.x]*y + ffFixedTerm_[i][threadIdx.x];
            }
        default:
            {
                using std::min;
                const auto i = (state-1)%4;
                const auto j = SubframeLabelManager::FullState(i);

                auto score = SubframeScore(i, data);

                // When signal is stronger than full-frame mean, ensure that
                // subframe density does not exceed full-frame density.
                const auto mu = ffmean_[i][threadIdx.x];
                const auto xmu = data * mu;
                const auto mumu = mu*mu;

                const auto y = data - mu;
                const auto fScore =  nhalfVal * y * pInvVar_[i][threadIdx.x]*y + ffFixedTerm_[i][threadIdx.x];

                // xmu > mumu means that the Euclidean projection of x onto mu is
                // greater than the norm of mu.
                const PBHalf2 one = PBHalf2(1.0f);
                const PBHalf2 zero = PBHalf2(0.0f);
                score = Blend(xmu > mumu,
                              min(min(score, fScore - one), zero),
                              score);

                return score;
            }
        }
    }

 private:
    __device__ void SubframeSetup(const LaneModelParameters& model)
    {
        static constexpr float pi_f = 3.1415926536f;
        const PBHalf2 sqrtHalfPi = PBHalf2(std::sqrt(0.5f * pi_f));
        const PBHalf2 halfVal = PBHalf2(0.5f);
        const PBHalf2 oneSixth = PBHalf2(1.0f / 6.0f);
        const PBHalf2 one = PBHalf2(1.0f);

        const auto& bMode = model.BaselineMode();
        const auto bMean = bMode.means[threadIdx.x];
        const auto bVar = bMode.vars[threadIdx.x];
        const auto bSigma = sqrt(bVar);

        // Low joint in the score function.
        x0_[threadIdx.x] = bMean + bSigma * sqrtHalfPi;
        bInvVar_[threadIdx.x] = one / bVar;

        #pragma unroll 1
        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto& dma = model.AnalogMode(a);
            const auto aMean = dma.means[threadIdx.x];
            logPMean_[a][threadIdx.x] = log(aMean - bMean);
            const auto aVar = dma.vars[threadIdx.x];
            const auto aSigma = sqrt(aVar);
            // High joint in the score function.
            x1_[a][threadIdx.x] = aMean - aSigma * sqrtHalfPi;
            pInvVar_[a][threadIdx.x] = one / aVar;
        }

        // Assume that the dimmest analog is the only one at risk for
        // violating x0_ <= x1_[a].
        {
            const auto& dm = model.AnalogMode(3);
            const auto aMean = dm.means[threadIdx.x];
            const auto aVar = dm.vars[threadIdx.x];
            // Edge-frame mean and variance.
            auto eMean = halfVal * (bMean + aMean);
            auto eVar = oneSixth * (aMean - bMean);
            eVar += bVar + aVar;
            eVar *= halfVal;
            // Normal approximation fallback when x1 < x0.
            dimFallback_.Mean(eMean).Variance(eVar);
        }
    }

    __device__ PBHalf2 SubframeScore(unsigned int a, PBHalf2 val) const
    {
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);

        // Score between x0 and x1.
        const auto lowExtrema = val < x0_[threadIdx.x];
        const auto cap = Blend(lowExtrema, x0_[threadIdx.x], x1_[a][threadIdx.x]);
        auto fixedTerm = Blend(lowExtrema, bInvVar_[threadIdx.x], pInvVar_[a][threadIdx.x]);
        fixedTerm = nhalfVal * fixedTerm;

        const auto extrema = lowExtrema || (val > x1_[a][threadIdx.x]);
        PBHalf2 s = Blend(extrema, fixedTerm * pow2(val - cap), PBHalf2(0.0f)) - logPMean_[a][threadIdx.x];

        if (a == numAnalogs-1)
        {
            // Fall back to normal approximation when x1 < x0.
            s = Blend(x0_[threadIdx.x] <= x1_[a][threadIdx.x], s, dimFallback_(val));
        }
        return s;
    }

    Row bgMean_;
    Utility::CudaArray<Row, numAnalogs> ffmean_; // full-frame states

    Row bInvVar_;
    Utility::CudaArray<Row, numAnalogs> pInvVar_;

    // -0.5 * log det(V) - log 2\pi.
    Row bgFixedTerm_;    // background (a.k.a. baseline)
    Utility::CudaArray<Row, numAnalogs> ffFixedTerm_;    // full-frame states

    // Log of displacement of pulse means from baseline mean.
    Utility::CudaArray<Row, numAnalogs> logPMean_;

    // Joints in the subframe score function.
    Row x0_;
    Utility::CudaArray<Row, numAnalogs> x1_;

    // Normal approximation fallback for dimmest analog subframe when x1 < x0.
    NormalLog<gpuLaneWidth> dimFallback_;
};

}}}

#endif //PACBIO_CUDA_SUBFRAME_SCORER_H
