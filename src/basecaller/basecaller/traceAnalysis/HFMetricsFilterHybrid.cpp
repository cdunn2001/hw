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

#include "HFMetricsFilterHybrid.h"

namespace PacBio::Mongo::Basecaller
{

void HFMetricsFilterHybrid::Configure(uint32_t sandwichTolerance,
                                      uint32_t framesPerHFMetricBlock,
                                      double frameRate,
                                      bool realtimeActivityLabels)
{
    HFMetricsFilterDevice::Configure(sandwichTolerance,
                                     framesPerHFMetricBlock,
                                     frameRate,
                                     realtimeActivityLabels);
}

std::unique_ptr<HFMetricsFilter::BasecallingMetricsBatchT>
HFMetricsFilterHybrid::Process(
            const PulseBatchT& pulseBatch,
            const Data::BaselinerMetrics& baselinerMetrics,
            const ModelsBatchT& models,
            const Data::FrameLabelerMetrics& flMetrics,
            const Data::PulseDetectorMetrics& pdMetrics)
{
    auto basecallingMetricsGpu = (*device_)(pulseBatch, baselinerMetrics, models, flMetrics, pdMetrics);
    auto basecallingMetricsCpu = (*host_)(pulseBatch, baselinerMetrics, models, flMetrics, pdMetrics);

    if (basecallingMetricsGpu && basecallingMetricsCpu)
    {
        DiffMetrics(*basecallingMetricsGpu, *basecallingMetricsCpu);
    }
    else if (basecallingMetricsGpu)
    {
        PBLOG_ERROR << "GPU returning non-null basecalling metrics but CPU returning null basecalling metrics";
    }
    else if (basecallingMetricsCpu)
    {
        PBLOG_ERROR << "CPU returning non-null basecalling metrics but GPU returning null basecalling metrics";
    }

    return basecallingMetricsCpu;
}

void HFMetricsFilterHybrid::DiffMetrics(const BasecallingMetricsBatchT& gpu, const BasecallingMetricsBatchT& cpu)
{
    for (size_t lane = 0; lane < gpu.Size(); lane++)
    {
        const auto& laneMetricsGpu = gpu.GetHostView()[lane];
        const auto& laneMetricsCpu = cpu.GetHostView()[lane];

        for (size_t zmw = 0; zmw < laneSize; zmw++)
        {
            size_t zmwNum = (lane * laneSize) + zmw;
            if (laneMetricsGpu.startFrame[zmw] != laneMetricsCpu.startFrame[zmw])
                PBLOG_ERROR << "zmw = " << zmwNum
                            << " GPU startFrame = " << laneMetricsGpu.startFrame[zmw]
                            << " CPU startFrame = " << laneMetricsCpu.startFrame[zmw];

            if (laneMetricsGpu.numFrames[zmw] != laneMetricsCpu.numFrames[zmw])
                PBLOG_ERROR << "zmw = " << zmwNum
                            << " GPU numFrames = " << laneMetricsGpu.numFrames[zmw]
                            << " CPU numFrames = " << laneMetricsCpu.numFrames[zmw];

            const auto& startFrame = laneMetricsCpu.startFrame[zmw];
            const auto endFrame = laneMetricsCpu.startFrame[zmw] + laneMetricsCpu.numFrames[zmw];

            if (laneMetricsGpu.activityLabel[zmw] != laneMetricsCpu.activityLabel[zmw])
                PBLOG_ERROR << "zmw = " << zmwNum << " frames = [" << startFrame << "," << endFrame << "]"
                            << " GPU activity label = " << static_cast<uint8_t>(laneMetricsGpu.activityLabel[zmw])
                            << " CPU activity label = " << static_cast<uint8_t>(laneMetricsCpu.activityLabel[zmw]);

            auto compareUIntMetrics = [&](uint16_t gpuVal, uint16_t cpuVal, const std::string& name)
            {
                if (gpuVal != cpuVal)
                {
                    PBLOG_ERROR << "zmw = " << zmwNum << " frames = [" << startFrame << "," << endFrame << "]"
                                << " GPU " << name << " = " << gpuVal << " CPU " << name << " = " << cpuVal;
                }
            };

            auto compareFloatMetrics = [&](float gpuVal, float cpuVal, const std::string& name)
            {
                int ulp = 2;
                bool almost_equal = std::fabs(gpuVal - cpuVal) <= std::numeric_limits<float>::epsilon() * std::fabs(gpuVal + cpuVal) * ulp
                                    || std::fabs(gpuVal - cpuVal) < std::numeric_limits<float>::min();
                if (!almost_equal)
                {
                    PBLOG_ERROR << "zmw = " << zmwNum << " frames = [" << startFrame << "," << endFrame << "]"
                                << " GPU " << name << " = " << gpuVal << " CPU " << name << " = " << cpuVal;
                }
            };

            compareUIntMetrics(laneMetricsGpu.numPulseFrames[zmw], laneMetricsCpu.numPulseFrames[zmw], "numPulseFrames");
            compareUIntMetrics(laneMetricsGpu.numBaseFrames[zmw], laneMetricsCpu.numBaseFrames[zmw], "numBaseFrames");
            compareUIntMetrics(laneMetricsGpu.numSandwiches[zmw], laneMetricsCpu.numSandwiches[zmw], "numSandwiches");
            compareUIntMetrics(laneMetricsGpu.numHalfSandwiches[zmw], laneMetricsCpu.numHalfSandwiches[zmw], "numHalfSandwiches");
            compareUIntMetrics(laneMetricsGpu.numPulseLabelStutters[zmw], laneMetricsCpu.numPulseLabelStutters[zmw], "numPulseLabelStutters");
            compareUIntMetrics(laneMetricsGpu.numBases[zmw], laneMetricsCpu.numBases[zmw], "numBases");
            compareUIntMetrics(laneMetricsGpu.numPulses[zmw], laneMetricsCpu.numPulses[zmw], "numPulses");
            compareUIntMetrics(laneMetricsGpu.numFramesBaseline[zmw], laneMetricsCpu.numFramesBaseline[zmw], "numFramesBaseline");

            for (size_t a = 0; a < numAnalogs; a++)
            {
                compareFloatMetrics(laneMetricsGpu.pkMidSignal[zmw][a], laneMetricsCpu.pkMidSignal[zmw][a], "pkMidSignal" + std::to_string(a));
                compareFloatMetrics(laneMetricsGpu.bpZvar[zmw][a], laneMetricsCpu.bpZvar[zmw][a], "bpZvar" + std::to_string(a));
                compareFloatMetrics(laneMetricsGpu.pkZvar[zmw][a], laneMetricsCpu.pkZvar[zmw][a], "pkZvar" + std::to_string(a));
                compareFloatMetrics(laneMetricsGpu.pkMax[zmw][a], laneMetricsCpu.pkMax[zmw][a], "pkMax" + std::to_string(a));

                compareUIntMetrics(laneMetricsGpu.numPkMidFrames[zmw][a], laneMetricsCpu.numPkMidFrames[zmw][a], "numPkMidFrames" + std::to_string(a));
                compareUIntMetrics(laneMetricsGpu.numPkMidBasesByAnalog[zmw][a], laneMetricsCpu.numPkMidBasesByAnalog[zmw][a], "numPkMidBasesByAnalog" + std::to_string(a));
                compareUIntMetrics(laneMetricsGpu.numBasesByAnalog[zmw][a], laneMetricsCpu.numBasesByAnalog[zmw][a], "numBasesByAnalog" + std::to_string(a));
                compareUIntMetrics(laneMetricsGpu.numPulsesByAnalog[zmw][a], laneMetricsCpu.numPulsesByAnalog[zmw][a], "numPulsesByAnalog" + std::to_string(a));
            }

            compareFloatMetrics(laneMetricsGpu.frameBaselineDWS[zmw], laneMetricsCpu.frameBaselineDWS[zmw], "frameBaselineDWS");
            compareFloatMetrics(laneMetricsGpu.frameBaselineVarianceDWS[zmw], laneMetricsCpu.frameBaselineVarianceDWS[zmw], "frameBaselineVarianceDWS");
            compareFloatMetrics(laneMetricsGpu.autocorrelation[zmw], laneMetricsCpu.autocorrelation[zmw], "autocorrelation");
            compareFloatMetrics(laneMetricsGpu.pulseDetectionScore[zmw], laneMetricsCpu.pulseDetectionScore[zmw], "pulseDetectionScore");
        }
    }
}

HFMetricsFilterHybrid::~HFMetricsFilterHybrid() = default;

} // PacBio::Mongo::Basecaller