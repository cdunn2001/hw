// Copyright (c) 2022, Pacific Biosciences of California, Inc.
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

#include "DmeEmHybrid.h"

#include <sstream>

#include <dataTypes/configs/BasecallerDmeConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Utility;
using namespace PacBio::Mongo::Data;

using FrameIntervalType = PacBio::Mongo::IntInterval<FrameIndexType>;


namespace PacBio {
namespace Mongo {
namespace Basecaller {

using FloatVec = LaneArray<float>;
using BoolVec = LaneMask<>;

// Static configuration parameters
static float rtol_ = -1.0;    // Relative tolerance
static float atol_ = -1.0;    // Absolute tolerance
static constexpr float epsHalf = 9.7656e-4f;    // "numeric_limits<half>::epsilon()"
 
void DmeEmHybrid::Configure(const Data::BasecallerDmeConfig &dmeConfig,
                          const Data::AnalysisConfig &analysisConfig)
{
    DmeEmDevice::Configure(dmeConfig, analysisConfig);
    DmeEmHost::Configure(dmeConfig, analysisConfig);

    // We assume that the results being compared go through a half-precision
    // representation.
    rtol_ = dmeConfig.HybridRtol * epsHalf;
    atol_ = dmeConfig.HybridAtol * epsHalf;
}

DmeEmHybrid::DmeEmHybrid(uint32_t poolId, unsigned int poolSize)
    : CoreDMEstimator(poolId, poolSize)
    , device_(std::make_unique<DmeEmDevice>(poolId, poolSize))
    , host_(std::make_unique<DmeEmHost>(poolId, poolSize))
{}

// In this diff function, the parameters go in the order of the decreasing precision
FloatVec AbsErr(const FloatVec& x0, const FloatVec& x1)
{
    return abs(x0 - x1);
}

// Relative tolerance is scaled by |x0|.
BoolVec AllClose(const FloatVec& x0, const FloatVec& x1)
{
    return AbsErr(x0, x1) <= (atol_ + rtol_ * abs(x0));
}

void ReportIfDiverged(const FloatVec& xcpu, const FloatVec& xgpu, size_t l, size_t a, const char* metricDesc)
{
    constexpr uint32_t colnum = 8;
    constexpr int fmtPrecision = 4;
    constexpr int fmtGap = 3;
    constexpr int fmtWidth = 6 + fmtPrecision + fmtGap;
    
    if (!all(AllClose(xcpu, xgpu)))
    {
        const auto cdata = MakeUnion(xcpu);
        const auto gdata = MakeUnion(xgpu);

        std::stringstream str;
        str << std::endl;
        str << metricDesc << " has diverged in " << "lane: " << l << ", pulse: " << a << std::endl;

        str << std::scientific;
        str.precision(fmtPrecision);

        for (uint32_t i = 0; i < laneSize; i += colnum)
        {
            str << "CPU(" << std::setw(2) << i << "):";
            for (uint32_t j = i; j < std::min(i+colnum, laneSize) ; j++)
            {
                str << std::setw(fmtWidth) << cdata[j];
            }
            str << std::endl;
            str << "GPU(" << std::setw(2) << i << "):";
            for (uint32_t j = i; j < std::min(i+colnum, laneSize) ; j++)
            {
                str << std::setw(fmtWidth) << gdata[j];
            }
            str << '\n' << std::endl;
        }

        PBLOG_ERROR << str.str();
    }
}


void DiffModels(const CoreDMEstimator::PoolDetModel& cpu, const CoreDMEstimator::PoolDetModel& gpu)
{
    using LaneDetModelHost = Data::DetectionModelHost<FloatVec>;

    // Verify the frame interval
    const auto cpufi = cpu.frameInterval;
    const auto gpufi = gpu.frameInterval;
    if (cpufi != gpufi)
    {
        std::stringstream str;
        str << "Frame interval is different:" << std::endl;
        str << "CPU FI LOW: " << cpufi.Lower() << ", HIGH: " << cpufi.Upper() << std::endl;
        str << "GPU FI LOW: " << gpufi.Lower() << ", HIGH: " << gpufi.Upper() << std::endl;
        PBLOG_ERROR << str.str();
    }

    const auto cpuldms = cpu.data.GetHostView();
    const auto gpuldms = gpu.data.GetHostView();
    if (cpuldms.Size() != gpuldms.Size())
    {
        std::stringstream str;
        str << "Pool size is different:" << std::endl;
        str << "CPU POOL SIZE: " << cpuldms.Size() << std::endl;
        str << "GPU POOL SIZE: " << gpuldms.Size() << std::endl;
        PBLOG_ERROR << str.str();
    }

    for (size_t l = 0; l < cpuldms.Size(); ++l)
    {
        const LaneDetModelHost cldm(cpuldms[l], cpufi);
        const LaneDetModelHost gldm(gpuldms[l], gpufi);
        const auto& cpuBgMode      = cldm.BaselineMode();
        const auto& gpuBgMode      = gldm.BaselineMode();
        ReportIfDiverged(cpuBgMode.SignalMean(),   gpuBgMode.SignalMean(),  l, 0, "BG mean");
        ReportIfDiverged(cpuBgMode.SignalCovar(),  gpuBgMode.SignalCovar(), l, 0, "BG var");
        ReportIfDiverged(cpuBgMode.Weight(),       gpuBgMode.Weight(),      l, 0, "BG weight");

        for (size_t i = 0; i < numAnalogs; ++i)
        {
            const auto& cpma = cldm.DetectionModes()[i];
            const auto& gpma = gldm.DetectionModes()[i];

            ReportIfDiverged(cpma.SignalMean(),   gpma.SignalMean(),  l, i, "Pulse mean");
            ReportIfDiverged(cpma.SignalCovar(),  gpma.SignalCovar(), l, i, "Pulse var");
            ReportIfDiverged(cpma.Weight(),       gpma.Weight(),      l, i, "Pulse weight");
        }
    }
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
