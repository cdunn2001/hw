#ifndef mongo_basecaller_traceAnalysis_DmeDiagnostics_H_
#define mongo_basecaller_traceAnalysis_DmeDiagnostics_H_

// Copyright (c) 2018-9, Pacific Biosciences of California, Inc.
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
//  Defines class template DmeDiagnostics.

#include <algorithm>
#include <bitset>
#include <cassert>

//#include <common/AlignedNew.h>
#include <common/AlignedVector.h>
#include <common/simd/SimdTypeTraits.h>

#include "MaxLikelihoodDiagnostics.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A simple struct representing the pertinent details for a goodness-of-fit test.
template <typename VF>
struct alignas(VF) GoodnessOfFitTest
{
    /// The test statistic (e.g., Pearson's chi-square or G-test).
    VF testStatistic;

    /// Degrees of freedom. Typically used as a parameter for the distribution
    /// of the test statistic.
    VF degOfFreedom;

    /// P-value of the value of the test statistic and degrees of freedom.
    VF pValue;
};


/// A collection of diagnostics that characterize an estimation of the
/// detection model parameters (a.k.a. DME).
/// \tparam VF SIMD floating-point type.
template <typename VF>
struct alignas(VF) DmeDiagnostics
{
public:     // Types
    using FloatVec = VF;

public:     // Static constants
    static constexpr auto vecSize = PacBio::Simd::SimdTypeTraits<VF>::width;

public: // Data
    /// Factors of estimation confidence. Each element should in [0, 1].
    AlignedVector<VF> confidFactors;

    /// Indicate frame range that produced the model for a given zmw.
    /// Vectorized since older models get carried forward if estimation for
    /// a given zmw fails.
    unsigned long startFrame {0};
    unsigned long stopFrame {0};

    /// Bit set indicating lane-level issues encountered during model estimation.
    unsigned short laneEventCode {0};

    /// Bit sets indicating ZMW-level issues encountered during model estimation.
    std::array<unsigned short, vecSize> zmwEventCode {};

    /// Indicates whether a full estimation was attempted in the detection
    /// model filter.
    bool fullEstimation {false};

    /// Diagnostics associated with the iterative maximum-likelihood algorithm.
    MaxLikelihoodDiagnostics<VF> mldx {};

    /// G-test for goodness of fit.
    GoodnessOfFitTest<VF> gTest {0.0f, 0.0f, 1.0f};

public:     // Structors
    DmeDiagnostics()
    {
        assert(std::count(zmwEventCode.cbegin(), zmwEventCode.cend(), 0) == static_cast<int64_t>(zmwEventCode.size()));
    }

public:     // Const functions
    /// Overall confidence score for the estimate of the detection model.
    /// Conventionally, 0 <= confidenceScore <= 1.
    /// 0 indicates failure, i.e., don't trust the estimate at all.
    VF Confidence() const
    {
        VF conf {1.0f};
        for (const auto& cf : confidFactors) conf *= cf;
        return conf;
    }
};

}}} // PacBio::Mongo::Basecaller

#endif  // mongo_basecaller_traceAnalysis_DmeDiagnostics_H_
