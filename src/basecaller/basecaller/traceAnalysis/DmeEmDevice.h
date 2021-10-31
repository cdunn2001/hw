#ifndef mongo_basecaller_traceAnalysis_DmeEmDevice_H_
#define mongo_basecaller_traceAnalysis_DmeEmDevice_H_

// Copyright (c) 2019-2021, Pacific Biosciences of California, Inc.
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

#include "CoreDMEstimator.h"

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// This class and the next could probably be unified with the Lane versions via
// a little template work, but first we need to iron out issues relating to
// how/if we share code between the host and the GPU.  If for example we created
// a LaneArray class that worked on the GPU, some of this complexity would maybe
// be easier to manange
struct ZmwAnalogMode
{
    template <int low>
    CUDA_ENABLED void Assign(const Data::LaneAnalogMode<Cuda::PBHalf2, 32>& mode, int idx)
    {
        mean = mode.means[idx].Get<low>();
        var = mode.vars[idx].Get<low>();
        weight = mode.weights[idx].Get<low>();
    }

    float mean;
    float var;
    float weight;
};

struct ZmwDetectionModel
{
    static constexpr int numAnalogs = 4;

    template <int low>
    CUDA_ENABLED void Assign(const Data::LaneModelParameters<Cuda::PBHalf2, 32>& model, int idx)
    {
        for (int i = 0; i < numAnalogs; ++i)
        {
            analogs[i].Assign<low>(model.AnalogMode(i), idx);
        }
        baseline.Assign<low>(model.BaselineMode(), idx);
    }

    Cuda::Utility::CudaArray<ZmwAnalogMode, numAnalogs> analogs;
    ZmwAnalogMode baseline;

    // TODO: Even the host version does not have a way to propogate the confidence between
    //       estimations
    float confidence = 0;
};

/// Implements DetectionModelEstimator using a Expectation-Maximization (EM)
/// approach for model estimation that runs on the CPU (as opposed to the GPU).
class DmeEmDevice : public CoreDMEstimator
{
public:     // Types
    using LaneDetModel = Data::LaneModelParameters<Cuda::PBHalf2, laneSize/2>;

public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::MovieConfig &movConfig);

public:
    DmeEmDevice(uint32_t poolId, unsigned int poolSize);

    PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const override;
private:    // Customized implementation
    void EstimateImpl(const PoolHist& hist,
                      PoolDetModel* detModel) const override;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DmeEmDevice_H_
