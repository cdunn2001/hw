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

#include <algorithm>
#include <common/MongoConstants.h>

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
    using LaneAnalogModeT = Data::LaneAnalogMode<Cuda::PBHalf2, laneSize/2>;

    template <int low>
    CUDA_ENABLED void Assign(const LaneAnalogModeT& mode, int idx)
    {
        static_assert(low == 0 || low == 1);
        assert(idx >= 0);
        assert(static_cast<unsigned int>(idx) < mode.means.size());
        mean = mode.means[idx].Get<low>();
        var = mode.vars[idx].Get<low>();
        weight = mode.weights[idx].Get<low>();
    }

    template <int low>
    CUDA_ENABLED void Export(int index, LaneAnalogModeT* mode)
    {
        static_assert(low == 0 || low == 1);
        assert(index >= 0);
        assert(static_cast<unsigned int>(index) < mode->weights.size());
        mode->weights[index].Set<low>(weight);
        mode->means[index].Set<low>(mean);

        // Guard against overflow converting from single- to half-precision.
        const auto cvar = std::min(var, 65000.0f);
        mode->vars[index].Set<low>(cvar);
    }

    float mean;
    float var;
    float weight;
};

struct FiTypeDevice
{
    int32_t lo;
    int32_t up;
};

struct ZmwDetectionModel
{
    static constexpr int numAnalogs = 4;
    using LaneModelParamsT = Data::LaneModelParameters<Cuda::PBHalf2, laneSize/2>;

    template <int low>
    CUDA_ENABLED void Assign(const LaneModelParamsT& model, int idx)
    {
        static_assert(low == 0 || low == 1);
        assert(idx >= 0);
        assert(static_cast<unsigned int>(idx) < model.Confidence().size());
        confidence = model.Confidence()[idx].Get<low>();
        for (int i = 0; i < numAnalogs; ++i)
        {
            analogs[i].Assign<low>(model.AnalogMode(i), idx);
        }
        baseline.Assign<low>(model.BaselineMode(), idx);
    }

    // Transcribe the values in *this to the (index, low) slot of *model.
    template <int low>
    CUDA_ENABLED void Export(int index, LaneModelParamsT* model)
    {
        static_assert(low == 0 || low == 1);
        assert(index >= 0);
        assert(static_cast<unsigned int>(index) < model->Confidence().size());
        model->Confidence()[index].Set<low>(confidence);
        baseline.Export<low>(index, &model->BaselineMode());
        for (unsigned int i = 0; i < numAnalogs; ++i)
        {
            analogs[i].Export<low>(index, &model->AnalogMode(i));
        }
    }

    // Transcribe the values in *this to the (index, low) slot of *model.
    CUDA_ENABLED void Export(int index, int low, LaneModelParamsT* model)
    {
        assert(low == 0 || low == 1);
        if (low == 0) this->Export<0>(index, model);
        else this->Export<1>(index, model);
    }

    Cuda::Utility::CudaArray<ZmwAnalogMode, numAnalogs> analogs;
    ZmwAnalogMode baseline;

    float confidence = 0;
};

/// Implements DetectionModelEstimator using a Expectation-Maximization (EM)
/// approach for model estimation that runs on the CPU (as opposed to the GPU).
class DmeEmDevice : public CoreDMEstimator
{
public:     // Static functions
    static void Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::AnalysisConfig &analysisConfig);

public:
    DmeEmDevice(uint32_t poolId, unsigned int poolSize);

    PoolDetModel InitDetectionModels(const PoolBaselineStats& blStats) const override;

private:    // Customized implementation
    void EstimateImpl(const PoolHist& hist,
                      const Data::BaselinerMetrics& metrics,
                      PoolDetModel* detModel) const override;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DmeEmDevice_H_
