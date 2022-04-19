// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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
//  Defines members of class DmeEmDevice.

#include "DmeEmHybrid.h"

#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo::Data;

using FrameIntervalType = PacBio::Mongo::IntInterval<FrameIndexType>;


namespace PacBio {
namespace Mongo {
namespace Basecaller {

void DmeEmHybrid::Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::AnalysisConfig &analysisConfig)
{
    DmeEmDevice::Configure(dmeConfig, analysisConfig);
    DmeEmHost::Configure(dmeConfig, analysisConfig);
}

DmeEmHybrid::DmeEmHybrid(uint32_t poolId, unsigned int poolSize)
    : CoreDMEstimator(poolId, poolSize)
    , device_(std::make_unique<DmeEmDevice>(poolId, poolSize))
    , host_(std::make_unique<DmeEmHost>(poolId, poolSize))
{}

void DiffModels(const CoreDMEstimator::PoolDetModel& cpu, const CoreDMEstimator::PoolDetModel& gpu)
{
}

CoreDMEstimator::PoolDetModel DmeEmHybrid::InitDetectionModels(const PoolBaselineStats& blStats) const
{
    auto hostModel = host_->InitDetectionModels(blStats);
    auto gpuModel = device_->InitDetectionModels(blStats);

    DiffModels(hostModel, gpuModel);
    return hostModel;
}

void DmeEmHybrid::EstimateImpl(const PoolHist& hist,
                             const Data::BaselinerMetrics& metrics,
                             PoolDetModel* detModel) const
{
    auto gpuModel = detModel->Copy();

    host_->Estimate(hist, metrics, detModel);
    device_->Estimate(hist, metrics, &gpuModel);

    DiffModels(*detModel, gpuModel);
}

}}}     // namespace PacBio::Mongo::Basecaller
