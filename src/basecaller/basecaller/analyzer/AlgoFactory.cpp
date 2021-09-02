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

#include "AlgoFactory.h"

#include <sstream>
#include <pacbio/PBException.h>

#include <basecaller/traceAnalysis/Baseliner.h>
#include <basecaller/traceAnalysis/BaselinerParams.h>
#include "basecaller/traceAnalysis/BaselineStatsAggregatorDevice.h"
#include <basecaller/traceAnalysis/BaselineStatsAggregatorHost.h>
#include <basecaller/traceAnalysis/CoreDMEstimator.h>
#include <basecaller/traceAnalysis/DetectionModelEstimator.h>
#include <basecaller/traceAnalysis/DmeEmHost.h>
#include <basecaller/traceAnalysis/DmeEmDevice.h>
#include <basecaller/traceAnalysis/DeviceHFMetricsFilter.h>
#include <basecaller/traceAnalysis/DeviceMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/DevicePulseAccumulator.h>
#include <basecaller/traceAnalysis/FrameLabeler.h>
#include <basecaller/traceAnalysis/FrameLabelerDevice.h>
#include <basecaller/traceAnalysis/FrameLabelerHost.h>
#include <basecaller/traceAnalysis/HFMetricsFilter.h>
#include <basecaller/traceAnalysis/HostHFMetricsFilter.h>
#include <basecaller/traceAnalysis/HostPulseAccumulator.h>
#include <basecaller/traceAnalysis/HostSimulatedPulseAccumulator.h>
#include <basecaller/traceAnalysis/HostMultiScaleBaseliner.h>
#include <basecaller/traceAnalysis/HostNoOpBaseliner.h>
#include <basecaller/traceAnalysis/PulseAccumulator.h>
#include <basecaller/traceAnalysis/SubframeLabelManager.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>
#include <basecaller/traceAnalysis/TraceHistogramAccumHost.h>
#include <basecaller/traceAnalysis/DeviceTraceHistogramAccum.h>

#include <dataTypes/configs/BasecallerAlgorithmConfig.h>
#include <dataTypes/configs/MovieConfig.h>

using std::make_unique;
using std::ostringstream;
using std::unique_ptr;

using namespace PacBio::Cuda::Memory;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

AlgoFactory::AlgoFactory(const Data::BasecallerAlgorithmConfig& bcConfig)
{
    // Baseliner
    baselinerOpt_ = bcConfig.baselinerConfig.Method;

    // Histogram accumulator
    histAccumOpt_ = bcConfig.traceHistogramConfig.Method;
    baselineStatsAggregatorOpt_ = bcConfig.baselineStatsAggregatorConfig.Method;

    // Detection model estimator
    dmeOpt_ = bcConfig.dmeConfig.Method;

    frameLabelerOpt_ = bcConfig.frameLabelerConfig.Method;
    pulseAccumOpt_ = bcConfig.pulseAccumConfig.Method;
    hfMetricsOpt_ = bcConfig.Metrics.Method;

    // TODO: Capture remaining options for algorithms.
}

AlgoFactory::~AlgoFactory()
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        HostNoOpBaseliner::Finalize();
        break;
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        DeviceMultiScaleBaseliner::Finalize();
        break;
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
        HostMultiScaleBaseliner::Finalize();
        break;
    default:
        PBLOG_ERROR << "Unrecognized method option for Baseliner: "
                    << baselinerOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
        break;
    }

    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::NoOp:
        FrameLabeler::Finalize();
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsDevice:
        FrameLabelerDevice::Finalize();
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsHost:
        FrameLabelerHost::Finalize();
        break;
    default:
        PBLOG_ERROR << "Unrecognized method option for FrameLabeler: "
                    << frameLabelerOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
        break;
    }

    switch (pulseAccumOpt_)
    {
    case Data::BasecallerPulseAccumConfig::MethodName::NoOp:
        PulseAccumulator::Finalize();
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostSimulatedPulses:
        HostSimulatedPulseAccumulator::Finalize();
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostPulses:
        HostPulseAccumulator<SubframeLabelManager>::Finalize();
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::GpuPulses:
        DevicePulseAccumulator<SubframeLabelManager>::Finalize();
        break;
    default:
        ostringstream msg;
        PBLOG_ERROR << "Unrecognized method option for PulseAccumulator: "
                    << pulseAccumOpt_.toString()
                    << ".  Should be impossible to see this message, constructor should have thrown";
        break;
    }

    switch (hfMetricsOpt_)
    {
    case Data::BasecallerMetricsConfig::MethodName::NoOp:
        NoHFMetricsFilter::Finalize();
        break;
    case Data::BasecallerMetricsConfig::MethodName::Host:
        HostHFMetricsFilter::Finalize();
        break;
    case Data::BasecallerMetricsConfig::MethodName::Gpu:
        DeviceHFMetricsFilter::Finalize();
        break;
    default:
        ostringstream msg;
        PBLOG_ERROR << "Unrecognized method option for HFMetricsFilter: "
                    << hfMetricsOpt_.toString()
                    << ".  Should be impossible to see this message, "
                    << "constructor should have thrown";
        break;
    }
}


void AlgoFactory::Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                            const Data::MovieConfig& movConfig)
{
    switch (baselinerOpt_)
    {
    case Data::BasecallerBaselinerConfig::MethodName::NoOp:
        HostNoOpBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
        DeviceMultiScaleBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
    case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
        HostMultiScaleBaseliner::Configure(bcConfig.baselinerConfig, movConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (histAccumOpt_)
    {
    case Data::BasecallerTraceHistogramConfig::MethodName::Host:
        TraceHistogramAccumHost::Configure(bcConfig.traceHistogramConfig);
        break;
    case Data::BasecallerTraceHistogramConfig::MethodName::Gpu:
        DeviceTraceHistogramAccum::Configure(bcConfig.traceHistogramConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for TraceHistogramAccumulator: " << histAccumOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    DetectionModelEstimator::Configure(bcConfig.dmeConfig);

    switch (dmeOpt_)
    {
    case Data::BasecallerDmeConfig::MethodName::Fixed:
    case Data::BasecallerDmeConfig::MethodName::EmHost:
        DmeEmHost::Configure(bcConfig.dmeConfig, movConfig);
        break;
    case Data::BasecallerDmeConfig::MethodName::EmDevice:
        DmeEmDevice::Configure(bcConfig.dmeConfig, movConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for CoreDMEstimator: "
            << dmeOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::NoOp:
        FrameLabeler::Configure();
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsDevice:
        FrameLabelerDevice::Configure(movConfig);
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsHost:
        FrameLabelerHost::Configure(movConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for FrameLabeler: " << frameLabelerOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (pulseAccumOpt_)
    {
    case Data::BasecallerPulseAccumConfig::MethodName::NoOp:
        PulseAccumulator::Configure(bcConfig.pulseAccumConfig);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostSimulatedPulses:
        HostSimulatedPulseAccumulator::Configure(bcConfig.pulseAccumConfig);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostPulses:
        HostPulseAccumulator<SubframeLabelManager>::Configure(
            movConfig,
            bcConfig.pulseAccumConfig);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::GpuPulses:
        DevicePulseAccumulator<SubframeLabelManager>::Configure(
            movConfig,
            bcConfig.pulseAccumConfig);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for pulseAccumulator: " << pulseAccumOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }

    switch (hfMetricsOpt_)
    {
    case Data::BasecallerMetricsConfig::MethodName::Gpu:
        DeviceHFMetricsFilter::Configure(bcConfig.Metrics.sandwichTolerance,
                                         bcConfig.Metrics.framesPerHFMetricBlock,
                                         movConfig.frameRate,
                                         bcConfig.Metrics.realtimeActivityLabels);
        break;
    default:
        HFMetricsFilter::Configure(bcConfig.Metrics.sandwichTolerance,
                                   bcConfig.Metrics.framesPerHFMetricBlock,
                                   movConfig.frameRate,
                                   bcConfig.Metrics.realtimeActivityLabels);
    }
}


unique_ptr<Baseliner>
AlgoFactory::CreateBaseliner(unsigned int poolId,
                             const Data::BatchDimensions& dims,
                             StashableAllocRegistrar& registrar) const
{
    const auto& params = FilterParamsLookup(baselinerOpt_);
    // TODO: We are currently overloading BasecallerBaselinerConfig::MethodName
    // to represent both the baseliner method and param. When the GPU version
    // is ready to take params, then the Configure() above should store the params
    // and a new baseliner config option should be for selecting the actual baseliner.
    switch (baselinerOpt_)
    {
        case Data::BasecallerBaselinerConfig::MethodName::NoOp:
            return std::make_unique<HostNoOpBaseliner>(poolId);
            break;
        case Data::BasecallerBaselinerConfig::MethodName::DeviceMultiScale:
            return std::make_unique<DeviceMultiScaleBaseliner>(poolId, dims.lanesPerBatch, params, &registrar);
            break;
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::MultiScaleSmall:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleLarge:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleMedium:
        case Data::BasecallerBaselinerConfig::MethodName::TwoScaleSmall:
            // TODO: scaler currently set to default 1.0f
            return std::make_unique<HostMultiScaleBaseliner>(poolId, 1.0f, params,
                                                             dims.lanesPerBatch);
            break;
        default:
            ostringstream msg;
            msg << "Unrecognized method option for Baseliner: " << baselinerOpt_.toString() << '.';
            throw PBException(msg.str());
            break;
    }
}

std::unique_ptr<FrameLabeler>
AlgoFactory::CreateFrameLabeler(unsigned int poolId,
                                const Data::BatchDimensions& dims,
                                StashableAllocRegistrar& registrar) const
{
    switch (frameLabelerOpt_)
    {
    case Data::BasecallerFrameLabelerConfig::MethodName::NoOp:
        return std::make_unique<FrameLabeler>(poolId);
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsDevice:
        return std::make_unique<FrameLabelerDevice>(poolId, dims.lanesPerBatch, &registrar);
        break;
    case Data::BasecallerFrameLabelerConfig::MethodName::SubFrameGaussCapsHost:
        return std::make_unique<FrameLabelerHost>(poolId, dims.lanesPerBatch);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for FrameLabeler: " << frameLabelerOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }
}

unique_ptr<TraceHistogramAccumulator>
AlgoFactory::CreateTraceHistAccumulator(unsigned int poolId, const Data::BatchDimensions& dims,
                                        StashableAllocRegistrar& registrar) const
{
    switch (histAccumOpt_)
    {
    case Data::BasecallerTraceHistogramConfig::MethodName::Host:
        return std::make_unique<TraceHistogramAccumHost>(poolId, dims.lanesPerBatch);
        break;
    case Data::BasecallerTraceHistogramConfig::MethodName::Gpu:
        return std::make_unique<DeviceTraceHistogramAccum>(poolId, dims.lanesPerBatch, &registrar);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for TraceHistogramAccumulator: "
            << histAccumOpt_ << '.';
        throw PBException(msg.str());
        break;
    }
}

unique_ptr<BaselineStatsAggregator>
AlgoFactory::CreateBaselineStatsAggregator(unsigned int poolId,
                                           const Data::BatchDimensions& dims,
                                           Cuda::Memory::StashableAllocRegistrar& registrar) const
{
    switch (baselineStatsAggregatorOpt_)
    {
    case Data::BasecallerBaselineStatsAggregatorConfig::MethodName::Host:
        return std::make_unique<BaselineStatsAggregatorHost>(poolId, dims.lanesPerBatch);
        break;
    case Data::BasecallerBaselineStatsAggregatorConfig::MethodName::Gpu:
        return std::make_unique<BaselineStatsAggregatorDevice>(poolId, dims.lanesPerBatch, &registrar);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for BaselineStatsAggregator: "
            << baselineStatsAggregatorOpt_ << '.';
        throw PBException(msg.str());
        break;
    }
}

std::unique_ptr<CoreDMEstimator>
AlgoFactory::CreateCoreDMEstimator(unsigned int poolId,
                                   const Data::BatchDimensions& dims,
                                   StashableAllocRegistrar&) const
{
    switch (dmeOpt_)
    {
    case Data::BasecallerDmeConfig::MethodName::Fixed:
    case Data::BasecallerDmeConfig::MethodName::EmHost:
        return make_unique<DmeEmHost>(poolId, dims.lanesPerBatch);

    case Data::BasecallerDmeConfig::MethodName::EmDevice:
        return make_unique<DmeEmDevice>(poolId, dims.lanesPerBatch);

    default:
        ostringstream msg;
        msg << "Unrecognized method option for CoreDMEstimator: "
            << dmeOpt_ << '.';
        throw PBException(msg.str());
        break;
    }
}

std::unique_ptr<PulseAccumulator>
AlgoFactory::CreatePulseAccumulator(unsigned int poolId,
                                    const Data::BatchDimensions& dims,
                                    StashableAllocRegistrar& registrar) const
{
    switch (pulseAccumOpt_)
    {
    case Data::BasecallerPulseAccumConfig::MethodName::NoOp:
        return std::make_unique<PulseAccumulator>(poolId);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostSimulatedPulses:
        return std::make_unique<HostSimulatedPulseAccumulator>(poolId, dims.lanesPerBatch);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::HostPulses:
        return std::make_unique<HostPulseAccumulator<SubframeLabelManager>>(poolId, dims.lanesPerBatch);
        break;
    case Data::BasecallerPulseAccumConfig::MethodName::GpuPulses:
        return std::make_unique<DevicePulseAccumulator<SubframeLabelManager>>(poolId, dims.lanesPerBatch, &registrar);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for pulseAccumulator: " << pulseAccumOpt_.toString() << '.';
        throw PBException(msg.str());
        break;
    }
}

std::unique_ptr<HFMetricsFilter>
AlgoFactory::CreateHFMetricsFilter(unsigned int poolId,
                                   const Data::BatchDimensions& dims,
                                   StashableAllocRegistrar& registrar) const
{
    switch (hfMetricsOpt_)
    {
    case Data::BasecallerMetricsConfig::MethodName::NoOp:
        return std::make_unique<NoHFMetricsFilter>(poolId);
        break;
    case Data::BasecallerMetricsConfig::MethodName::Host:
        return std::make_unique<HostHFMetricsFilter>(poolId, dims.lanesPerBatch);
        break;
    case Data::BasecallerMetricsConfig::MethodName::Gpu:
        return std::make_unique<DeviceHFMetricsFilter>(poolId, dims.lanesPerBatch, &registrar);
        break;
    default:
        ostringstream msg;
        msg << "Unrecognized method option for HFMetricsFilter: " << hfMetricsOpt_.toString() << '.';
        throw PBException(msg.str());
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
