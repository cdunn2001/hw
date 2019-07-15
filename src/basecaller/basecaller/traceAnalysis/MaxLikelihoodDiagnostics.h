#ifndef mongo_basecaller_traceAnalysis_MaxLikelihoodDiagnostics_H_
#define mongo_basecaller_traceAnalysis_MaxLikelihoodDiagnostics_H_

// Copyright (c) 2017-2019, Pacific Biosciences of California, Inc.
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
//  Defines stuct MaxLikelihoodDiagnostics.

#include <common/simd/SimdConvTraits.h>
#include <common/simd/SimdTypeTraits.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A simple type that bundles a few diagnostics for iterative
/// maximum-likelihood estimation methods, including the expectation-
/// maximization-like algorithms used in DmeMonochrome.
template <typename VF>
struct alignas(VF) MaxLikelihoodDiagnostics
{
    using VB = Simd::BoolConv<VF>;
    using VI = Simd::IndexConv<VF>;

    MaxLikelihoodDiagnostics()
        : converged     {false}
        , iterCount     {0}
        , degOfFreedom  {-1}
        , logLike       {-std::numeric_limits<float>::infinity()}
        , deltaLogLike  {-std::numeric_limits<float>::infinity()}
        , goodnessOfFit {std::numeric_limits<float>::quiet_NaN()}
    { }

    /// Did the iterative estimation algorithm converge for a particular unit cell?
    VB converged;

    /// For each unit cell, the number of iterations performed to achieve
    /// convergence, or 0 if convergence was not achieved.
    /// iterCount >= 0.
    VI iterCount;

    /// Degrees of freedom for each unit cell.
    /// Number of data minus number of free model parameters.
    VI degOfFreedom;

    /// For each unit cell, the largest log likelihood that was achieved.
    VF logLike;

    /// For each unit cell, the change in log likelihood in the iteration that
    /// achieved the largest log likelihood.
    VF deltaLogLike;

    /// For each unit cell, a goodness-of-fit statistic.
    /// Definition depends on the implementation of the fitting algorithm.
    VF goodnessOfFit;

    /// Set the \c converged flag for elements indicated by \a aConverged and
    /// and record the iteration count, log likehood, and delta log likelihood
    /// for those elements.
    void Converged(const VB& aConverged,
                   const unsigned int aIter,
                   const VF& aLogLike,
                   const VF& aDeltaLogLike,
                   const VF& aGoodnessOfFit = VF(0.0f))
    {
        iterCount = Blend(aConverged & !converged, VI(aIter), iterCount);
        logLike = Blend(aConverged, aLogLike, logLike);
        deltaLogLike = Blend(aConverged, aDeltaLogLike, deltaLogLike);
        goodnessOfFit = Blend(aConverged, aGoodnessOfFit, goodnessOfFit);
        converged |= aConverged;
    }

    /// Update the log liklihood and delta log likelihood for any element where
    /// we've not already fully converged.  \a softConverged is used to indicate
    /// lanes where we want to record our progress but have not necessarily
    /// fully converged, and \a hardConverged is used to indicate lanes that
    /// have completed
    void TieredConverged(const VB& hardConverged,
                         const VB& softConverged,
                         const VB& failedZmw,
                         const unsigned int aIter,
                         const VF& aLogLike,
                         const VF& aDeltaLogLike)
    {
        assert(none(hardConverged & converged));
        assert(none(softConverged & converged));
        assert(none(softConverged & failedZmw));
        assert(none(hardConverged & failedZmw));

        // Update likelihoods every iteration until full convergence (unless
        // the lane failed as cans happen in binned dme)
        logLike = Blend(!(converged | failedZmw), aLogLike, logLike);
        deltaLogLike = Blend(!(converged | failedZmw), aDeltaLogLike, deltaLogLike);

        iterCount = Blend(softConverged, aIter, iterCount);
        converged |= hardConverged;
    }
};

}}}     // PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_MaxLikelihoodDiagnostics_H_
