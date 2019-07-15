#ifndef mongo_basecaller_traceAnalysis_DmeEmHost_H_
#define mongo_basecaller_traceAnalysis_DmeEmHost_H_

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
//  Defines class DmeEmHost.

#include <common/LaneArray.h>
#include <dataTypes/DetectionModelHost.h>
#include <dataTypes/UHistogramSimd.h>

#include "DetectionModelEstimator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// Implements DetectionModelEstimator using a Expectation-Maximization (EM)
/// approach for model estimation.
class DmeEmHost : public DetectionModelEstimator
{
public:     // Types
    using FloatVec = LaneArray<float>;
    using IntVec = LaneArray<int>;
    using BoolVec = LaneMask<>;
    using CountVec = LaneArray<LaneHist::CountType>;
    using UHistType = Data::UHistogramSimd<FloatVec, CountVec>;
    using LaneDetModelHost = Data::DetectionModelHost<FloatVec>;

    enum ZmwStatus : uint16_t
    {
        OK = 0,
        NO_CONVERGE     = 1u << 0,
        INSUF_DATA      = 1u << 1,
        VLOW_SIGNAL     = 1u << 2
    };

public:     // Static constants
    /// Number of free model parameters.
    /// Five mixture fractions, background mean, background variance, and
    /// pulse amplitude scale.
    static constexpr unsigned short nModelParams = 8;

    /// Minimum number of frames required for parameter estimation.
    static constexpr unsigned int nFramesMin = 20 * nModelParams;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::MovieConfig &movConfig);

    // If mask[i], a[i] |= bits.
    static void SetBits(const BoolVec& mask, int32_t bits, IntVec* a)
    {
        // TODO: There is probably a more efficient way to implement this.
        const IntVec b = *a | IntVec(bits);
        *a = Blend(mask, b, *a);
    }

public:
    DmeEmHost(uint32_t poolId, unsigned int poolSize);

private:    // Types
    using LaneHistSimd = Data::UHistogramSimd<typename LaneHist::DataType, typename LaneHist::CountType>;

private:    // Customized implementation
    void EstimateImpl(const PoolHist& hist, PoolDetModel* detModel) const override;

private:    // Static data
    static unsigned short emIterLimit_;
    static bool iterToLimit_;
    static float pulseAmpRegCoeff_;

private:    // Static functions
    static FloatVec PrelimScaleFactor(const LaneDetModelHost& model, const UHistType& hist);

private:    // Functions
    void EstimateLaneDetModel(const UHistType& hist, LaneDetModelHost* detModel) const;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DmeEmHost_H_
