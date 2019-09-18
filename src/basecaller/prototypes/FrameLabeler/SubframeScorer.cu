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

#include "SubframeScorer.cuh"

using namespace PacBio::Cuda::Utility;

namespace PacBio {
namespace Cuda {
namespace Subframe {


__device__ TransitionMatrix::TransitionMatrix(Utility::CudaArray<AnalogMeta, numAnalogs> meta)
{
    half zero = __float2half(0.0f);
    for (int i = 0; i < numStates; ++i)
    {
        for (int j = 0; j < numStates; ++j)
        {
            data_[i][j] = zero;
        }
    }
    // Algorithm courtesy Rob Grothe.

    // Avoid singularities in the computations to follow.  Could probably
    // find a new analytic form without singularities, but just avoid
    // ratios of 1.0 for now.
    for (size_t i = 0; i < numAnalogs; ++i)
    {
        static constexpr float delta = 0.01f;
        if (std::abs(meta[i].ipdSSRatio - 1.0f) < delta)
        {
            meta[i].pwSSRatio = (meta[i].pwSSRatio < 1.0f) ? 1.0f - delta : 1.0f + delta;
        }
        if (std::abs(meta[i].ipdSSRatio - 1.0f) < delta)
        {
            meta[i].ipdSSRatio = (meta[i].ipdSSRatio < 1.0f) ? 1.0f - delta : 1.0f + delta;
        }
    }

    // Mean IPD for the input analog set.
    const float meanIpd = [&meta](){
        float sum = 0.0f;
        for (uint32_t i = 0; i < numAnalogs; ++i) sum+=meta[i].ipd;
        return sum/numAnalogs;
    }();

    // Given that an analog, with an average width and two-step ratio,
    // what are the probabilities that we detect a 1, 2 or 3 frame pulse.
    // See analytic work done in SPI-1674 and particularly subtask
    // SPI-1754
    auto computeLenProbs = [&](float width, float ratio) -> CudaArray<float, 3>
    {
        // Make sure r is small instead of 0, to avoid indeterminate forms
        // in the math below.
        ratio = std::max(ratio, 1e-6f);

        const auto k1 = (1.0f + ratio) / width;
        const auto k2 = (1.0f + 1.0f / ratio) / width;
        const auto dk = k1 - k2;

        const auto c1 = 1.0f - std::exp(-k1);
        const auto c2 = 1.0f - std::exp(-k2);

        CudaArray<float, 3> ret;
        ret[0] = 1 - k1 * c2 / k2 / dk + k2 * c1 / k1 / dk;
        ret[1] = k1 * (c2 * c2) / k2 / dk - k2 * (c1 * c1) / k1 / dk;
        ret[2] = k1 * (c2 * c2) * (1 - c2) / k2 / dk - k2 * (c1 * c1) * (1 - c1) / k1 / dk;
        return ret;
    };

    // Slurp in fudge factors from the config file
    const auto alphaFactor = 1.0f;//config.Alpha;
    const auto betaFactor = 1.0f;//config.Beta;
    const auto gammaFactor = 1.0f;//config.Gamma;

    // Precompute the probabilities of 1, 2 and 3 frame ipds after each analog
    CudaArray<CudaArray<float, 3>, numAnalogs> ipdLenProbs;
    ipdLenProbs[0] = computeLenProbs(meta[0].ipd, meta[0].ipdSSRatio);
    ipdLenProbs[1] = computeLenProbs(meta[1].ipd, meta[1].ipdSSRatio);
    ipdLenProbs[2] = computeLenProbs(meta[2].ipd, meta[2].ipdSSRatio);
    ipdLenProbs[3] = computeLenProbs(meta[3].ipd, meta[3].ipdSSRatio);

    // alpha[i] is probability of going from the i down to another up state.
    // The direct down-to-full is currently disallowed, so we roll both
    // the 1 and 2 frame ipd detection probabilities into here.  Factor
    // of 0.25 is to account that each of the 4 analogs are equally likely
    // when coming out of a down state.
    CudaArray<float, numAnalogs> alpha;
    alpha[0] = alphaFactor * 0.25f * (ipdLenProbs[0][0] + ipdLenProbs[0][1]);
    alpha[1] = alphaFactor * 0.25f * (ipdLenProbs[1][0] + ipdLenProbs[1][1]);
    alpha[2] = alphaFactor * 0.25f * (ipdLenProbs[2][0] + ipdLenProbs[2][1]);
    alpha[3] = alphaFactor * 0.25f * (ipdLenProbs[3][0] + ipdLenProbs[3][1]);
    const auto& alphaSum = [&]() {
        float sum = 0.0f;
        for (uint32_t i = 0; i < numAnalogs; ++i)
        {
            sum += alpha[i];
        }
        return sum;
    }();

    // If we're passing through a full baseline state, the transition to
    // a new pulse will not know what previous pulse it came from.  In this
    // case use an averaged IPD, with a ratio of 0 (since it could potentially
    // be a long IPD)
    CudaArray<float, 3> meanIpdLenProbs = computeLenProbs(meanIpd, 0.0f);

    // Probability of 3 *or more* frame IPD. Since the d->u transition is
    // up to 2 frames, this is also the probability of entering the full
    // baseline state (given that a pulse has ended)
    const auto meanIpd3m = 1 - meanIpdLenProbs[0] - meanIpdLenProbs[1];

    // an IPD of 3 means we went d->0->u'.  Thus the 0->u probability is
    // that, given that we've passed through the full 0 state (e.g. meanIpd3m)
    const auto pulseStartProb = meanIpdLenProbs[2] / meanIpd3m;

    data_[0][0] = 1.0f - pulseStartProb;

    for (uint32_t i = 0; i < numAnalogs; ++i)
    {
        const auto full = SubframeLabelManager::FullState(i);
        const auto up = SubframeLabelManager::UpState(i);
        const auto down = SubframeLabelManager::DownState(i);

        // probabilities of 1,2 and 3 frame pulses for this analog
        const auto& analogLenProbs = computeLenProbs(meta[i].pw, meta[i].pwSSRatio);

        // Probability of a 1 frame pulse is the sum of P(u->0) and all of P(u->u')
        // The former is gamma and the later is alpha. So summing over alpha
        // and solving for gamma gives us:
        const auto gamma = gammaFactor * analogLenProbs[0] / (1 + alphaSum);

        // Fudged probability of 2 frame pulse
        const auto beta = betaFactor * analogLenProbs[1];

        // N.B. For the below, remember that the matrix is indexed as
        // transScore_(current, previous).  Transitions read reversed
        // from what one might expect!

        // Give each analog equal probability to start
        data_[up][0] = 0.25f * pulseStartProb;

        // Record exactly 1 and 2 frame pulses.
        data_[0][up] = gamma;
        data_[down][up] = beta;

        // Normalize by subtracting out all alternative paths:
        // gamma            -- u -> 0
        // gamma * alphaSum -- u->u' (summed over all u')
        // beta             -- u -> d
        data_[full][up] = 1 - gamma - gamma * alphaSum - beta;

        // Probability that pulse is 3 or more frames long.  Subsequently
        // also the probability that we at least pass through the full frame
        // state.
        const auto prob3ms = 1.0f - analogLenProbs[0] - analogLenProbs[1];

        // analogLenProbs[2] = u->T->d path.  The T->d transition is that
        // probability *given* we've passed through the T state (e.g. prob3sum)
        data_[down][full] = analogLenProbs[2] / prob3ms;

        // Normalize by subtracting out the (only) alternative path
        data_[full][full] = __float2half(1.0f) - data_[down][full];

        // Again normalize by subtracting out the alternatives, which are
        // all the alpha terms that let us go from d->u'
        data_[0][down] = 1 - alphaSum;

        // Handle the dense section of u->u' and d->u' transitions
        for (uint32_t j = 0; j < numAnalogs; j++)
        {
            auto upPrime = SubframeLabelManager::UpState(j);

            // Alpha already defined as 1 or 2 frame ipd event
            // (no full baseline frame)
            data_[upPrime][down] = alpha[j];

            // To go handle u->u' we also need to include the probability
            // of a 1 frame pulse before the 1 or 2 frame ipd.
            data_[upPrime][up] = gamma * alpha[j];
        }
    }

    // Convert to log-prob scores
    for (int i = 0; i < numStates; ++i)
    {
        for (int j = 0; j < numStates; ++j)
        {
            data_[i][j] = hlog(data_[i][j]);
        }
    }
}

}}}
