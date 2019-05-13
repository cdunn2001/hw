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
#include <common/cuda/utility/CudaArray.cuh>

#include "AnalogModel.cuh"
#include "AnalogMeta.h"

namespace PacBio {
namespace Cuda {
namespace Subframe {

static constexpr int numAnalogs = 4;
static constexpr int numStates = 13;

__device__ inline constexpr int FullState(int i) { return i+1; }
__device__ inline constexpr int UpState(int i) { return i+1 + numAnalogs; }
__device__ inline constexpr int DownState(int i) { return i+1 + 2*numAnalogs; }

struct __align__(128) TransitionMatrix
{
    using T = half;
    using Row = Utility::CudaArray<T, numStates>;

    // Initializes the cuda matrix on the device.  Should be invoked by only
    // a single thread.
    __device__ TransitionMatrix(Utility::CudaArray<AnalogMeta, numAnalogs> meta);

    __device__ T operator()(int row, int col) const { return data_[row][col]; }
    __device__ T Entry(int row, int col)      const { return data_[row][col]; }

private:
    Utility::CudaArray<Row, numStates> data_;
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
    { return 0.5f / scaleFactor_[threadIdx.x]; }

    /// Sets Variance to new value, which must be positive.
    /// \returns *this.
    __device__ NormalLog& Variance(const PBHalf2& value)
    {
        //assert(all(value > 0.0f));
        normTerm_[threadIdx.x] = PBHalf2(-0.5f) * (PBHalf2(log2pi_f) + log(value));
        scaleFactor_[threadIdx.x] = PBHalf2(0.5f) / value;
        return *this;
    }

public:
    /// Evaluate the distribution at \a x.
    __device__ PBHalf2 operator()(const PBHalf2& x) const
    { return normTerm_[threadIdx.x] - pow2(x - mean_[threadIdx.x]) * scaleFactor_[threadIdx.x]; }

private:    // Data
    Row mean_;
    Row normTerm_;
    Row scaleFactor_;
};

// Gause caps edge frame scorer.  This is a near unchanged transcription
// of what exists in Sequel.  All operations are designed to be done by cuda blocks
// with one thread per pair of zmw.
template <size_t gpuLaneWidth>
struct __align__(128) GauseCapsScorer
{
    static constexpr unsigned int numAnalogs = 4;
    using Row = Utility::CudaArray<PBHalf2, gpuLaneWidth>;

    GauseCapsScorer() = default;

    __device__ void Setup(const LaneModelParameters<gpuLaneWidth>& model)
    {
        static constexpr float pi_f = 3.1415926536f;
        const PBHalf2 sqrtHalfPi = PBHalf2(std::sqrt(0.5f * pi_f));
        const PBHalf2 halfVal = PBHalf2(0.5f);
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 oneSixth = PBHalf2(1.0f / 6.0f);

        const auto& bMode = model.BaselineMode();
        const auto bMean = bMode.means[threadIdx.x];
        const auto bVar = bMode.vars[threadIdx.x];
        const auto bSigma = sqrt(bVar);

        // Low joint in the score function.
        x0_[threadIdx.x] = bMean + bSigma * sqrtHalfPi;
        bFixedTerm_[threadIdx.x] = nhalfVal / bVar;

        for (unsigned int a = 0; a < numAnalogs; ++a)
        {
            const auto& dma = model.AnalogMode(a);
            const auto aMean = dma.means[threadIdx.x];
            logPMean_[a][threadIdx.x] = log(aMean - bMean);
            const auto aVar = dma.vars[threadIdx.x];
            const auto aSigma = sqrt(aVar);
            // High joint in the score function.
            x1_[a][threadIdx.x] = aMean - aSigma * sqrtHalfPi;
            pFixedTerm_[a][threadIdx.x] = nhalfVal / aVar;
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

    __device__ PBHalf2 operator()(unsigned int a, PBHalf2 val) const
    {
        // Score between x0 and x1.
        const auto lowExtrema = val < x0_[threadIdx.x];
        const auto cap = Blend(lowExtrema, x0_[threadIdx.x], x1_[a][threadIdx.x]);
        const auto fixedTerm = Blend(lowExtrema, bFixedTerm_[threadIdx.x], pFixedTerm_[a][threadIdx.x]);

        const auto extrema = lowExtrema || (val > x1_[a][threadIdx.x]);
        PBHalf2 s = Blend(extrema, fixedTerm * pow2(cap - cap), PBHalf2(0.0f)) - logPMean_[a][threadIdx.x];

        if (a == numAnalogs-1)
        {
            // Fall back to normal approximation when x1 < x0.
            s = Blend(x0_[threadIdx.x] <= x1_[a][threadIdx.x], s, dimFallback_(val));
        }
        return s;
    }
 private:
    // Log of displacement of pulse means from baseline mean.
    Utility::CudaArray<Row, numAnalogs> logPMean_;

    // Joints in the score function.
    Row x0_;
    Utility::CudaArray<Row, numAnalogs> x1_;

    Row bFixedTerm_;
    Utility::CudaArray<Row, numAnalogs> pFixedTerm_;

    // Normal approximation fallback for dimmest analog when x1 < x0.
    NormalLog<gpuLaneWidth> dimFallback_;
};

// Subframe scorer using gause caps. This is a near unchanged transcription
// of what exists in Sequel, which is why the gause caps portion is
// in a separate class instead of inline.  All operations are designed to be
// done by cuda blocks with one thread per pair of zmw.
template <size_t gpuLaneWidth>
struct __align__(128) BlockStateScorer
{
    using Row = Utility::CudaArray<PBHalf2, gpuLaneWidth>;

    // Default constructor only exists to facilitate a shared memory instance.
    // The object is not in a valid state until after calling `Setup`
    BlockStateScorer() = default;

    __device__ void Setup(const LaneModelParameters<gpuLaneWidth>& model)
    {
        static constexpr float log2pi_f = 1.8378770664f;
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 one = PBHalf2(1.0f);

        sfScore_.Setup(model);

        const PBHalf2 normConst = PBHalf2(log2pi_f / 2.0f);

        // Background
        const auto& bgMode = model.BaselineMode();

        // Put -0.5 * log(det(V)) into bgFixedTerm_; compute bgInvCov_.
        // sym2x2MatrixInverse returns the determinant and writes the inverse
        // into the second argument.
        bgInvCov_[threadIdx.x] = one / bgMode.vars[threadIdx.x];
        bgFixedTerm_[threadIdx.x] = nhalfVal * log(bgInvCov_[threadIdx.x]) - normConst;
        bgMean_[threadIdx.x] = bgMode.means[threadIdx.x];

        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            // Full-frame states
            const auto& aMode = model.AnalogMode(i);
            ffFixedTerm_[i][threadIdx.x] = nhalfVal * log(one / aMode.vars[threadIdx.x]) - normConst;
            ffmean_[i][threadIdx.x] = aMode.means[threadIdx.x];
        }
    }

    __device__ Utility::CudaArray<PBHalf2, numStates> StateScores(PBHalf2 data) const
    {
        Utility::CudaArray<PBHalf2, numStates> score;
        const PBHalf2 halfVal = PBHalf2(0.5f);
        const PBHalf2 nhalfVal = PBHalf2(-0.5f);
        const PBHalf2 one = PBHalf2(1.0f);
        const PBHalf2 zero = PBHalf2(0.0f);

        {   // Compute the score for the background state.
            const auto mu = bgMean_[threadIdx.x];
            const auto y = data - mu;
            score[0] = nhalfVal * y * bgInvCov_[threadIdx.x] * y + bgFixedTerm_[threadIdx.x];
            // By definition, the ROI trace value for background is 0.
        }

        // Compute scores for full-frame pulse states.
        for (int i = 0; i < numAnalogs; ++i)
        {
            const auto j = FullState(i);
            const auto mu = ffmean_[i][threadIdx.x];
            const auto y = data - mu;
            score[j] = nhalfVal * y * ffInvCov_[i][threadIdx.x]*y + ffFixedTerm_[i][threadIdx.x];
            //score[j] += trc_.Roi(j, frame); // ROI exclusion
        }

        // Compute scores for subframe states.
        for (int i = 0; i < numAnalogs; ++i)
        {
            using std::min;
            const auto j = UpState(i);
            const auto k = DownState(i);
            const auto ff = FullState(i);

            score[j] = sfScore_(i, data);

            //score[j] += trc_.Roi(ff, frame);

            // When signal is stronger than full-frame mean, ensure that
            // subframe density does not exceed full-frame density.
            const auto mu = ffmean_[i][threadIdx.x];
            const auto xmu = data * mu;
            const auto mumu = mu*mu;
            // xmu > mumu means that the Euclidean projection of x onto mu is
            // greater than the norm of mu.
            score[j] = Blend(xmu > mumu,
                             min(min(score[j], score[ff] - one), zero),
                             score[j]);

            // Could apply a similar constraint on the other side to prevent
            // large negative spikes from getting labelled as subframe instead
            // of baseline.

            // "Down" states assigned same score as "up" states.
            score[k] = score[j];
        }

        return score;
    }

    // Pre-computed inverse covariances.
    Row bgInvCov_;    // background (a.k.a. baseline)
    Row bgMean_;
    Utility::CudaArray<Row, numAnalogs> ffInvCov_; // full-frame states
    Utility::CudaArray<Row, numAnalogs> ffmean_; // full-frame states

    // The terms in the state scores that do not depend explicitly on the trace.
    // -0.5 * log det(V) - log 2\pi.
    Row bgFixedTerm_;    // background (a.k.a. baseline)
    Utility::CudaArray<Row, numAnalogs> ffFixedTerm_;    // full-frame states

    GauseCapsScorer<gpuLaneWidth> sfScore_;
};

}}}

#endif //PACBIO_CUDA_SUBFRAME_SCORER_H
