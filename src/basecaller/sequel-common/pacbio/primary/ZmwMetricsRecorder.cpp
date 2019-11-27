// Copyright (c) 2017, Pacific Biosciences of California, Inc.
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
// Description:
/// \brief definitions for a class for recording metrics per ZMW



#include <boost/numeric/conversion/cast.hpp>

#include <pacbio/ipc/Message.h>
#include <pacbio/logging/Logger.h>

#include <pacbio/primary/SequelDefinitions.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/ZmwMetricsRecorder.h>
#include <pacbio/primary/PrimaryConfig.h>
#include "ZmwMetricsRecorder.h"

using boost::numeric_cast;

namespace PacBio {
namespace Primary {

template <typename TMetrics>
const std::string ZmwMetricsRecorder<TMetrics>::MetricsSuperChunk::metricsInfoStr = R"json(
{
"BaseRate": {
    "desc" : "Rate of base - incorporation events from sequencing ZMWs",
    "units": "1/sec"
},
"BaseWidth": {
    "desc" : "Base width - incorporation events from sequencing ZMWs",
    "units": "sec"
},
"PulseRate": {
    "desc": "Pulse rate from sequencing ZMWs",
    "units": "1/sec"
},
"PulseWidth": {
    "desc" : "Pulse width - incorporation events from sequencing ZMWs",
    "units": "sec"
},
"Baselines": {
    "BaselineLevel_NoZMWsNoApertures": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red] from NoZMWsNoApertures ZMWs",
    "units": "photo e-"
    },
    "BaselineLevel_NoZMWsAperturesOpen": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red] from NoZMWsAperturesOpen ZMWs",
    "units": "photo e-"
    },
    "BaselineLevel": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red] from sequencing ZMWs",
        "units": "photo e-"
    },
    "BaselineStd": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red] from sequencing ZMWs",
        "units": "photo e-"
    },
    "BaselineLevel_TopStrip": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineStd_TopStrip": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineLevel_MidStrip": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineStd_MidStrip": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineLevel_BotStrip": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineStd_BotStrip": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineLevel_MidHigh": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineStd_MidHigh": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineLevel_TopHigh": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineStd_TopHigh": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineLevel_BotHigh": {
    "desc": "Background signal estimate ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    },
    "BaselineStd_BotHigh": {
    "desc": "Background noise (standard deviation) ordered by increasing wavelength [green, red]",
        "units": "photo e-"
    }
},
"Pkmids": {
    "PkmidSignal": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T] from sequencing ZMWs",
    "units": "photo e-"
    },
    "Pkmid_TopStrip": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T]",
    "units": "photo e-"
    },
    "Pkmid_MidStrip": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T]",
    "units": "photo e-"
    },
    "Pkmid_BotStrip": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T]",
    "units": "photo e-"
    },
    "Pkmid_TopHigh": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T]",
    "units": "photo e-"
    },
    "Pkmid_MidHigh": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T]",
    "units": "photo e-"
    },
    "Pkmid_BotHigh": {
    "desc": "The mean (excluding edge frames) intra-pulse DWS signal of pulses that are called as bases by analog channel [A,C,G,T]",
    "units": "photo e-"
    }
},
"SNR": {
    "SNR": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T] from sequencing ZMWs"
    },
    "SNR_TopStrip": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T]"
    },
    "SNR_MidStrip": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T]"
    },
    "SNR_BotStrip": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T]"
    },
    "SNR_TopHigh": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T]"
    },
    "SNR_MidHigh": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T]"
    },
    "SNR_BotHigh": {
    "desc": "Signal-to-Noise Ratio from pulses called as bases by analog channel [A,C,G,T]"
    }
}
}
)json";

template <typename TMetrics>
std::string ZmwMetricsRecorder<TMetrics>::ZmwGroupingNew::ToCSV() const
{
    std::ostringstream csvOutput;

    for (const auto& metric: rtRegion_->MetricsToReport())
    {
        RTMetricsRegionNew::zmwMetric m = RTMetricsRegionNew::zmwMetric::fromString(metric);
        switch (m)
        {
            case RTMetricsRegionNew::zmwMetric::Baseline:
                csvOutput << ArrayToCSV(baseline_);
                break;
            case RTMetricsRegionNew::zmwMetric::BaselineStd:
                csvOutput << ArrayToCSV(baselineStdDev_);
                break;
            case RTMetricsRegionNew::zmwMetric::Pkmid:
                csvOutput << ArrayToCSV(pkmid_);
                break;
            case RTMetricsRegionNew::zmwMetric::Snr:
                csvOutput << ArrayToCSV(snr_);
                break;
            case RTMetricsRegionNew::zmwMetric::BaseRate:
                csvOutput << ArrayToCSV(baseRate_);
                break;
            case RTMetricsRegionNew::zmwMetric::BaseWidth:
                csvOutput << ArrayToCSV(baseWidth_);
                break;
            case RTMetricsRegionNew::zmwMetric::PulseRate:
                csvOutput << ArrayToCSV(pulseRate_);
                break;
            case RTMetricsRegionNew::zmwMetric::PulseWidth:
                csvOutput << ArrayToCSV(pulseWidth_);
                break;
            default:
                throw PBException("Unknown reported metric type!");
        }
        csvOutput << ",";
    }

    return csvOutput.str();
}


template <typename TMetrics>
Json::Value ZmwMetricsRecorder<TMetrics>::ZmwGroupingNew::ToJson() const
{
    Json::Value metrics(Json::arrayValue);
    for (const auto& metric: rtRegion_->MetricsToReport())
    {
        Json::Value val;
        RTMetricsRegionNew::zmwMetric m = RTMetricsRegionNew::zmwMetric::fromString(metric);
        switch (m)
        {
            case RTMetricsRegionNew::zmwMetric::Baseline:
                val = ArrayToJson(baseline_);
                break;
            case RTMetricsRegionNew::zmwMetric::BaselineStd:
                val = ArrayToJson(baselineStdDev_);
                break;
            case RTMetricsRegionNew::zmwMetric::Pkmid:
                val = ArrayToJson(pkmid_);
                break;
            case RTMetricsRegionNew::zmwMetric::Snr:
                val = ArrayToJson(snr_);
                break;
            case RTMetricsRegionNew::zmwMetric::BaseRate:
                val = ArrayToJson(baseRate_);
                break;
            case RTMetricsRegionNew::zmwMetric::BaseWidth:
                val = ArrayToJson(baseWidth_);
                break;
            case RTMetricsRegionNew::zmwMetric::PulseRate:
                val = ArrayToJson(pulseRate_);
                break;
            case RTMetricsRegionNew::zmwMetric::PulseWidth:
                val = ArrayToJson(pulseWidth_);
                break;
            default:
                throw PBException("Unknown reported metric type!");
        }
        val["name"] = metric;
        metrics.append(val);
    }

    Json::Value root;
    root["region"] = this->Name();
    root["metrics"] = metrics;

    return root;
}

template <typename TMetrics>
bool ZmwMetricsRecorder<TMetrics>::ZmwGroupingNew::AddZmw(const PacBio::Primary::UnitCell& position,
                                                          const TMetrics& bm, bool isSequencingZmw)
{
    if (!ContainsZMW(static_cast<uint16_t>(position.x), static_cast<uint16_t>(position.y)))
        return false;

    // Check sampling stride after checking if region contains ZMW.
    numZmws_++;
    if (numZmws_ % rtRegion_->SamplingFactorStride() != 0)
        return false;

    // Track the total number of ZMWs considered for this region
    // prior to applying any filtering.
    totalZmws_++;

    // Report baseline and baseline sigma without checking if the ZMW is "sequencing".
    const auto& fbl = bm.TraceMetrics().FrameBaselineDWS();
    const auto& fstd = bm.TraceMetrics().FrameBaselineSigmaDWS();
    const auto& nf = bm.TraceMetrics().NumFramesBaseline();

    bool dmeSuccess = bm.TraceMetrics().FullEstimationAttempted() &&
                      bm.TraceMetrics().ModelUpdated() &&
                      bm.TraceMetrics().ConfidenceScore() > rtConfig_->minConfidenceScore;

    // Channel metrics
    for (size_t c = 0; c < NCam; c++)
    {
        switch (rtConfig_->baselineMode)
        {
            case RTMetricsConfig::BaselineMode::MODE2:
            {
                if (nf[c] > rtConfig_->minBaselineFrames && dmeSuccess)
                {
                    if (!isnan(fbl[c])) baseline_[c].Values()(fbl[c]);
                    if (!isnan(fstd[c])) baselineStdDev_[c].Values()(fstd[c]);
                }
                break;
            }
            case RTMetricsConfig::BaselineMode::MODE1:
            {
                if (nf[c] > rtConfig_->minBaselineFrames)
                {
                    if (!isnan(fbl[c])) baseline_[c].Values()(fbl[c]);
                    if (!isnan(fstd[c])) baselineStdDev_[c].Values()(fstd[c]);
                }
                break;
            }
            case RTMetricsConfig::BaselineMode::MODE0:
            default:
            {
                if (!isnan(fbl[c])) baseline_[c].Values()(fbl[c]);
                if (!isnan(fstd[c])) baselineStdDev_[c].Values()(fstd[c]);
                break;
            }
        }
    }

    // Metrics below are only reported if the ZMW is deemed "sequencing".
    if (!isSequencingZmw)
        return false;

    baseRate_[0].Values()((float) bm.NumBases() / (bm.TraceMetrics().NumFrames() / rtConfig_->frameRate));
    baseWidth_[0].Values()(bm.BaseWidth() / rtConfig_->frameRate);
    pulseRate_[0].Values()((float) bm.NumPulses() / (bm.TraceMetrics().NumFrames() / rtConfig_->frameRate));
    pulseWidth_[0].Values()((float) bm.PulseWidth() / rtConfig_->frameRate);

    const auto& pkmid = bm.PkmidMean();
    const auto& fsnr = bm.FrameSnr();
    std::array<uint16_t,4> nFrames;

    if (NCam == 1)
        std::fill(nFrames.begin(), nFrames.end(), nf.front());
    else
        nFrames = {{ nf[1], nf[1], nf[0], nf[0] }};

    // Analog metrics
    for (size_t a = 0; a < NAnalogs; a++)
    {
        switch (rtConfig_->signalMode)
        {
            case RTMetricsConfig::SignalMode::MODE2:
            {
                if (nFrames[a] > rtConfig_->minBaselineFrames && dmeSuccess)
                {
                    if (!isnan(pkmid[a])) pkmid_[a].Values()(pkmid[a]);
                    if (!isnan(fsnr[a])) snr_[a].Values()(fsnr[a]);
                }
                break;
            }
            case RTMetricsConfig::SignalMode::MODE1:
            {
                if (nFrames[a] > rtConfig_->minBaselineFrames)
                {
                    if (!isnan(pkmid[a])) pkmid_[a].Values()(pkmid[a]);
                    if (!isnan(fsnr[a])) snr_[a].Values()(fsnr[a]);
                }
                break;
            }
            case RTMetricsConfig::SignalMode::MODE0:
            default:
            {
                /// We don't check for DME success for the pkmid which leads this to be inconsistent
                /// with the SNR.
                if (!isnan(pkmid[a])) pkmid_[a].Values()(pkmid[a]);
                if (dmeSuccess && !isnan(fsnr[a])) snr_[a].Values()(fsnr[a]);
                break;
            }
        }
    }

    return true;
}


template <typename TMetrics>
bool ZmwMetricsRecorder<TMetrics>::ZmwGrouping::AddZmw(const PacBio::Primary::UnitCell& position,
                                                       const TMetrics& bm)
{
    if (!ContainsZMW(static_cast<uint16_t>(position.x), static_cast<uint16_t>(position.y)))
        return false;

    baseRate_.Values()((float) bm.NumBases() / (bm.TraceMetrics().NumFrames() / rtConfig_->frameRate));
    baseWidth_.Values()(bm.BaseWidth() / rtConfig_->frameRate);
    pulseRate_.Values()((float) bm.NumPulses() / (bm.TraceMetrics().NumFrames() / rtConfig_->frameRate));
    pulseWidth_.Values()((float) bm.PulseWidth() / rtConfig_->frameRate);

    const auto& fbl = bm.TraceMetrics().FrameBaselineDWS();
    const auto& fstd = bm.TraceMetrics().FrameBaselineSigmaDWS();
    const auto& nf = bm.TraceMetrics().NumFramesBaseline();

    bool dmeSuccess = bm.TraceMetrics().FullEstimationAttempted() &&
                      bm.TraceMetrics().ModelUpdated() &&
                      bm.TraceMetrics().ConfidenceScore() > rtConfig_->minConfidenceScore;

    // Channel metrics
    for (size_t c = 0; c < NCam; c++)
    {
        switch (rtConfig_->baselineMode)
        {
            case RTMetricsConfig::BaselineMode::MODE2:
            {
                if (nf[c] > rtConfig_->minBaselineFrames && dmeSuccess)
                {
                    if (!isnan(fbl[c])) baseline_[c].Values()(fbl[c]);
                    if (!isnan(fstd[c])) baselineStdDev_[c].Values()(fstd[c]);
                }
                break;
            }
            case RTMetricsConfig::BaselineMode::MODE1:
            {
                if (nf[c] > rtConfig_->minBaselineFrames)
                {
                    if (!isnan(fbl[c])) baseline_[c].Values()(fbl[c]);
                    if (!isnan(fstd[c])) baselineStdDev_[c].Values()(fstd[c]);
                }
                break;
            }
            case RTMetricsConfig::BaselineMode::MODE0:
            default:
            {
                /// We don't check for DME success as the original behavior was to report
                /// baseline metrics for all blocks. This leads to the baseline metrics
                /// being inconsistent with the SNR.
                if (!isnan(fbl[c])) baseline_[c].Values()(fbl[c]);
                if (!isnan(fstd[c])) baselineStdDev_[c].Values()(fstd[c]);
                break;
            }
        }
    }

    const auto& pkmid = bm.PkmidMean();
    const auto& fsnr = bm.FrameSnr();
    std::array<uint16_t,4> nFrames;

    if (NCam == 1)
        std::fill(nFrames.begin(), nFrames.end(), nf.front());
    else
        nFrames = {{ nf[1], nf[1], nf[0], nf[0] }};

    // Analog metrics
    for (size_t a = 0; a < NAnalogs; a++)
    {
        switch (rtConfig_->baselineMode)
        {
            case RTMetricsConfig::BaselineMode::MODE2:
            {
                if (nFrames[a] > rtConfig_->minBaselineFrames && dmeSuccess)
                {
                    if (!isnan(pkmid[a])) pkmid_[a].Values()(pkmid[a]);
                    if (!isnan(fsnr[a])) snr_[a].Values()(fsnr[a]);
                }
                break;
            }
            case RTMetricsConfig::BaselineMode::MODE1:
            {
                if (nFrames[a] > rtConfig_->minBaselineFrames)
                {
                    if (!isnan(pkmid[a])) pkmid_[a].Values()(pkmid[a]);
                    if (!isnan(fsnr[a])) snr_[a].Values()(fsnr[a]);
                }
                break;
            }
            case RTMetricsConfig::BaselineMode::MODE0:
            default:
            {
                /// We don't check for DME success for the pkmid which leads this to be inconsistent
                /// with the SNR.
                if (!isnan(pkmid[a])) pkmid_[a].Values()(pkmid[a]);
                if (dmeSuccess && !isnan(fsnr[a])) snr_[a].Values()(fsnr[a]);
                break;
            }
        }
    }

    return true;
}

template <typename TMetrics>
void ZmwMetricsRecorder<TMetrics>::MetricsSuperChunk::MetricsBlock::ComputeTimeStamps(uint32_t superChunkIndex,
                                                                                      uint64_t chunkTimeStampStart,
                                                                                      uint64_t timeStampDelta)
{
    // Scale start frame relative to super chunk.
    uint32_t startFrameRelativeToSuperChunk =
            startFrame_ - (superChunkIndex * GetPrimaryConfig().cache.framesPerSuperchunk);
    startFrameTimeStamp_ = chunkTimeStampStart + (startFrameRelativeToSuperChunk * timeStampDelta);
    endFrameTimeStamp_ = startFrameTimeStamp_ + (numFrames_ * timeStampDelta);
}

template <typename TMetrics>
void ZmwMetricsRecorder<TMetrics>::MetricsSuperChunk::MetricsBlock::AddZmw(
        const TMetrics& bm,
        const PacBio::Primary::UnitCell& position,
        bool isSequencingZmw)
{
    if (rtConfig_->newJsonFormat)
    {
        for (auto& r : regionsNew_)
            r.AddZmw(position, bm, isSequencingZmw);
    }
    else
    {
        numZmws_++;
        if (numZmws_ % rtConfig_->samplingFactorStride == 0)
        {
            noZMWsnoAperturesZMWs_.AddZmw(position, bm);
            noZMWsAperturesOpenZMWs_.AddZmw(position, bm);
            if (isSequencingZmw)
            {
                sequencingZMWs_.AddZmw(position, bm);
                for (auto& r : regions_)
                    r.AddZmw(position, bm);
            }
        }
    }
}

template <typename TMetrics>
Json::Value ZmwMetricsRecorder<TMetrics>::MetricsSuperChunk::ToJson() const
{
    Json::Value root;

    root["startFrameTimeStamp"] = timeStampStart_;
    root["frameTimeStampDelta"] = timeStampDelta_;

    Json::Value metricsChunk;
    Json::Value metricsBlocks(Json::arrayValue);

    if (rtConfig_->newJsonFormat)
    {
        for (size_t nm = 0; nm < numMetrics_; nm++)
        {
            Json::Value mb;
            mb["startFrame"] = metricsBlocks_[nm].StartFrame();
            mb["numFrames"] = metricsBlocks_[nm].NumFrames();
            mb["startFrameTimeStamp"] = metricsBlocks_[nm].StartFrameTimeStamp();
            mb["endFrameTimeStamp"] = metricsBlocks_[nm].EndFrameTimeStamp();

            Json::Value groups(Json::arrayValue);
            for (const auto& r : metricsBlocks_[nm].regionsNew_)
            {
                groups.append(r.ToJson());
            }
            mb["groups"] = groups;

            metricsBlocks.append(mb);
        }
    }
    else
    {
        Json::Value metricsInfo;
        Json::Reader reader;
        bool success = reader.parse(metricsInfoStr.c_str(), metricsInfo);
        if (!success)
        {
            throw PBException("failed to parse " + metricsInfoStr);
        }
        metricsChunk["metricsInfo"] = metricsInfo;

        for (size_t nm = 0; nm < numMetrics_; nm++)
        {
            Json::Value mb;
            mb["startFrame"] = metricsBlocks_[nm].StartFrame();
            mb["numFrames"] = metricsBlocks_[nm].NumFrames();
            mb["startFrameTimeStamp"] = metricsBlocks_[nm].StartFrameTimeStamp();
            mb["endFrameTimeStamp"] = metricsBlocks_[nm].EndFrameTimeStamp();

            Json::Value metrics;

            metrics["PulseRate"] = metricsBlocks_[nm].sequencingZMWs_.pulseRate_.ToJson(rtConfig_->minSampleSize);
            metrics["PulseWidth"] = metricsBlocks_[nm].sequencingZMWs_.pulseWidth_.ToJson(rtConfig_->minSampleSize);
            metrics["BaseRate"] = metricsBlocks_[nm].sequencingZMWs_.baseRate_.ToJson(rtConfig_->minSampleSize);
            metrics["BaseWidth"] = metricsBlocks_[nm].sequencingZMWs_.baseWidth_.ToJson(rtConfig_->minSampleSize);

            Json::Value baselines;
            Json::Value pkmids;
            Json::Value snr;

            baselines["BaselineLevel"] = metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToJson(metricsBlocks_[nm].sequencingZMWs_.baseline_, true);
            baselines["BaselineStd"] = metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToJson(metricsBlocks_[nm].sequencingZMWs_.baselineStdDev_, true);

            pkmids["PkmidSignal"] = metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToJson(metricsBlocks_[nm].sequencingZMWs_.pkmid_);
            snr["SNR"] = metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToJson(metricsBlocks_[nm].sequencingZMWs_.snr_);

            baselines["BaselineLevel_" + metricsBlocks_[nm].noZMWsnoAperturesZMWs_.Name()] =
                    metricsBlocks_[nm].noZMWsnoAperturesZMWs_.
                            ArrayToJson(metricsBlocks_[nm].noZMWsnoAperturesZMWs_.baseline_, true);
            baselines["BaselineLevel_" + metricsBlocks_[nm].noZMWsAperturesOpenZMWs_.Name()] =
                    metricsBlocks_[nm].noZMWsAperturesOpenZMWs_.
                            ArrayToJson(metricsBlocks_[nm].noZMWsAperturesOpenZMWs_.baseline_, true);

            for (const auto& r : metricsBlocks_[nm].regions_)
            {
                baselines["BaselineLevel_" + r.Name()] = r.ArrayToJson(r.baseline_, true);
                baselines["BaselineStd_" + r.Name()] = r.ArrayToJson(r.baselineStdDev_, true);
                pkmids["Pkmid_" + r.Name()] = r.ArrayToJson(r.pkmid_);
                snr["SNR_" + r.Name()] = r.ArrayToJson(r.snr_);
            }

            metrics["Baselines"] = baselines;
            metrics["Pkmids"] = pkmids;
            metrics["SNR"] = snr;

            mb["metrics"] = metrics;

            metricsBlocks.append(mb);
        }
    }

    metricsChunk["metricsBlocks"] = metricsBlocks;
    metricsChunk["numMetricsBlocks"] = numMetrics_;
    root["metricsChunk"] = metricsChunk;
    root["token"] = token_;

    return root;
}

template <typename TMetrics>
std::string ZmwMetricsRecorder<TMetrics>::MetricsSuperChunk::ToCSV() const
{
    std::ostringstream csvOutput;

    for (size_t nm = 0; nm < numMetrics_; nm++)
    {
        csvOutput << metricsBlocks_[nm].startFrame_ << ","
                  << metricsBlocks_[nm].numFrames_ << ","
                  << metricsBlocks_[nm].startFrameTimeStamp_ << ","
                  << metricsBlocks_[nm].endFrameTimeStamp_ << ",";

        if (rtConfig_->newJsonFormat)
        {
            for (const auto& r : metricsBlocks_[nm].regionsNew_)
            {
                csvOutput << r.ToCSV();
            }
            csvOutput << "NA" << std::endl;
        }
        else
        {
            csvOutput << metricsBlocks_[nm].sequencingZMWs_.pulseRate_.ToCSV(rtConfig_->minSampleSize) << ",";
            csvOutput << metricsBlocks_[nm].sequencingZMWs_.pulseWidth_.ToCSV(rtConfig_->minSampleSize) << ",";
            csvOutput << metricsBlocks_[nm].sequencingZMWs_.baseRate_.ToCSV(rtConfig_->minSampleSize) << ",";
            csvOutput << metricsBlocks_[nm].sequencingZMWs_.baseWidth_.ToCSV(rtConfig_->minSampleSize) << ",";

            csvOutput << metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToCSV(metricsBlocks_[nm].sequencingZMWs_.baseline_) << ",";
            csvOutput << metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToCSV(metricsBlocks_[nm].sequencingZMWs_.baselineStdDev_) << ",";

            csvOutput << metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToCSV(metricsBlocks_[nm].sequencingZMWs_.pkmid_) << ",";
            csvOutput << metricsBlocks_[nm].sequencingZMWs_.
                    ArrayToCSV(metricsBlocks_[nm].sequencingZMWs_.snr_) << ",";

            csvOutput << metricsBlocks_[nm].noZMWsnoAperturesZMWs_.
                    ArrayToCSV(metricsBlocks_[nm].noZMWsnoAperturesZMWs_.baseline_) << ",";
            csvOutput << metricsBlocks_[nm].noZMWsAperturesOpenZMWs_.
                    ArrayToCSV(metricsBlocks_[nm].noZMWsAperturesOpenZMWs_.baseline_) << ",";

            for (const auto& r : metricsBlocks_[nm].regions_)
            {
                csvOutput << r.ArrayToCSV(r.baseline_) << ",";
                csvOutput << r.ArrayToCSV(r.baselineStdDev_) << ",";
                csvOutput << r.ArrayToCSV(r.pkmid_) << ",";
                csvOutput << r.ArrayToCSV(r.snr_) << ",";
            }
            csvOutput << "NA" << std::endl;
        }
    }

    return csvOutput.str();
}

template <typename TMetrics>
std::string ZmwMetricsRecorder<TMetrics>::CSVHeader() const
{
    const auto& sampleStr = [](const std::string& v) {
        const std::array<std::string, 4> sHeaders = {{"Mean", "Median", "CV", "Size"}};
        std::array<std::string, 4> sH;
        for (size_t s = 0; s < sHeaders.size(); s++) sH[s] = v + "_" + sHeaders[s];
        return sH;
    };

    const auto& newSampleStr = [](const std::string& v) {
        const std::array<std::string, 5> sHeaders = {{"Mean", "Median", "CV", "Size", "Total"}};
        std::array<std::string, 5> sH;
        for (size_t s = 0; s < sHeaders.size(); s++) sH[s] = v + "_" + sHeaders[s];
        return sH;
    };

    const auto& baselineStr = [](const std::string& v) {
        const std::array<std::string, 2> bHeaders = {{"G", "R"}};
        std::array<std::string, NCam> bH;
        if (bH.size() == 2)
            for (size_t b = 0; b < bH.size(); b++) bH[b] = v + "_" + bHeaders[b];
        else
            for (size_t b = 0; b < bH.size(); b++) bH[b] = v;
        return bH;
    };

    const auto& analogStr = [](const std::string& v) {
        const std::array<std::string, 4> aHeaders = {{"A", "C", "G", "T"}};
        std::array<std::string, 4> aH;
        for (size_t a = 0; a < aHeaders.size(); a++) aH[a] = v + "_" + aHeaders[a];
        return aH;
    };

    std::ostringstream csvHeader;

    csvHeader << "StartFrame,NumFrames,StartFrameTS,EndFrameTS,";

    if (rtConfig_->newJsonFormat)
    {
        for (const auto& r : zmwRTRegions_)
        {
            for (const auto& m : r.MetricsToReport())
            {
                std::string regMet = r.Name() + "_" + m;
                RTMetricsRegionNew::zmwMetric metric = RTMetricsRegionNew::zmwMetric::fromString(m);
                switch (metric)
                {
                    case RTMetricsRegionNew::zmwMetric::Baseline:
                    case RTMetricsRegionNew::zmwMetric::BaselineStd:
                        for (const auto& b : baselineStr(regMet))
                            for (const auto& s : newSampleStr(b))
                                csvHeader <<  s << ",";
                        break;
                    case RTMetricsRegionNew::zmwMetric::Pkmid:
                    case RTMetricsRegionNew::zmwMetric::Snr:
                        for (const auto& a : analogStr(regMet))
                            for (const auto& s : newSampleStr(a))
                                csvHeader << s << ",";
                        break;
                    case RTMetricsRegionNew::zmwMetric::BaseRate:
                    case RTMetricsRegionNew::zmwMetric::BaseWidth:
                    case RTMetricsRegionNew::zmwMetric::PulseRate:
                    case RTMetricsRegionNew::zmwMetric::PulseWidth:
                        for (const auto& s : newSampleStr(regMet))
                            csvHeader << s << ",";
                        break;
                    default:
                        throw PBException("Unknown reported metric type!");
                }
            }
        }
        csvHeader << "End";
    }
    else
    {
        for (const auto& k : std::array<std::string, 4>({{"PulseRate", "PulseWidth", "BaseRate", "BaseWidth"}}))
            for (const auto& s : sampleStr(k))
                csvHeader << s << ",";

        for (const auto& c : std::array<std::string, 2>({{"BaselineLevel", "BaselineStd"}}))
            for (const auto& b : baselineStr(c))
                for (const auto& s : sampleStr(b))
                    csvHeader << s << ",";

        for (const auto& b : std::array<std::string, 2>({{"Pkmid", "SNR"}}))
            for (const auto& a : analogStr(b))
                for (const auto& s : sampleStr(a))
                    csvHeader << s << ",";

        for (const auto& c : std::array<std::string, 2>({{"NoZMWsNoApertures", "NoZMWsAperturesOpen"}}))
            for (const auto& b : baselineStr(c))
                for (const auto& s : sampleStr(b))
                    csvHeader << s << ",";

        for (const auto& r : regions_)
        {
            for (const auto& c : std::array<std::string, 2>({{"BaselineLevel", "BaselineStd"}}))
                for (const auto& b : baselineStr(r.name() + "_" + c))
                    for (const auto& s : sampleStr(b))
                        csvHeader << s << ",";

            for (const auto& b : std::array<std::string, 2>({{"Pkmid", "SNR"}}))
                for (const auto& a : analogStr(r.name() + "_" + b))
                    for (const auto& s : sampleStr(a))
                        csvHeader << s << ",";
        }
        csvHeader << "End";
    }

    return csvHeader.str();
}

template <typename TMetrics>
ZmwMetricsRecorder<TMetrics>::ZmwMetricsRecorder(const RTMetricsConfig& config,
                                                 double frameRate,
                                                 const std::vector<
                                                         std::pair<
                                                                 PacBio::Primary::ChipLayout::UnitFeature,
                                                                 PacBio::Primary::UnitCell>>& unitCellTypes,
                                                 const std::string& chipLayoutName,
                                                 const std::string& metricsFile,
                                                 const size_t maxSamples,
                                                 const std::string& token)
    : metricsPublisher_(nullptr)
    , rtConfig_(new ZmwMetricsRecorderConfig(config, frameRate, unitCellTypes, chipLayoutName, metricsFile))
    , maxSamples_(maxSamples)
    , token_(token)
{
    if (rtConfig_->newJsonFormat)
    {
        for (const auto& r : config.newRegions)
            zmwRTRegions_.emplace_back(r, rtConfig_->chipLayout.get());
    }
    else
    {
        for (const auto& r : config.regions)
            regions_.append(r);
    }

    metricsOut_.open(rtConfig_->metricsFile, std::ios_base::trunc);
    metricsOut_ << CSVHeader() << std::endl;

    StartMetricsThread();
}

template <typename TMetrics>
ZmwMetricsRecorder<TMetrics>::ZmwMetricsRecorder(PacBio::IPC::MessageSocketPublisher *metricsPublisher,
                                                 const Json::Value &setupJson,
                                                 const RTMetricsConfig& config,
                                                 const size_t maxSamples,
                                                 const std::string& token)
    : metricsPublisher_(metricsPublisher)
    , rtConfig_(new ZmwMetricsRecorderConfig(config, setupJson))
    , maxSamples_(maxSamples)
    , numMetricsSuperChunks_(config.numMetricsSuperChunks())
    , token_(token)
{
    if (rtConfig_->newJsonFormat)
    {
        for (const auto& r : config.newRegions)
            zmwRTRegions_.emplace_back(r, rtConfig_->chipLayout.get());
    }
    else
    {
        for (const auto& r : config.regions)
            regions_.append(r);
    }

    StartMetricsThread();
}

template <typename TMetrics>
void ZmwMetricsRecorder<TMetrics>::StartMetricsThread()
{
    // We don't know yet how many zmw will be used.  We'll start at 1000, but
    // quickly grow to the maximum as determined by the hardware in use
    PBLOG_INFO << "Configuring " << numMetricsSuperChunks_ << " MetricsSuperChunkBuffers";
    if (!rtConfig_->chipLayout) throw PBException("bad assumption that chipLayout would be created by now...");
    for(uint32_t i = 0; i < numMetricsSuperChunks_; i++)
    {
        auto x = std::unique_ptr<MetricsSuperChunkBuffer>(
                new MetricsSuperChunkBuffer(1000,
                                            rtConfig_->chipLayout->GetNumUnitCells(),
                                            GetPrimaryConfig().cache.blocksPerTranche));
        x->Reset();
        idleMetricsSuperChunkBufferQueue_.Push(std::move(x));
    }

    currentMetricsSuperChunkBuffer_ = idleMetricsSuperChunkBufferQueue_.Pop();
    sendMetricsThreadContinue_ = true;
    sendMetricsThread_ = std::thread([this]() {
        try
        {
            ComputeMetrics();
        }
        catch (std::runtime_error &err)
        {
            PBLOG_ERROR << "Exception caught in ComputeMetrics:" << err.what();
            // TODO: Indicate that we could not send metrics and raise the appropriate alarm.
        }
    });
}


template <typename TMetrics>
bool ZmwMetricsRecorder<TMetrics>::IsSequencingZmw(const MemoryBufferView<TMetrics>& metrics)
{
    if (rtConfig_->useRealtimeActivityLabels && metrics.size() > 0)
    {
        int numSingle = 0;
        int numMulti = 0;
        for (size_t i = 0; i < metrics.size(); ++i)
        {
            switch (metrics[i].ActivityLabel())
            {
                case ActivityLabeler::HQRFPhysicalState::SINGLE:
                    ++numSingle;
                    break;
                case ActivityLabeler::HQRFPhysicalState::MULTI:
                    ++numMulti;
                    break;
                default:
                    ; // Do nothing.
            }
        }
        // This is a simple initial heuristic: we only return true if there are
        // no multiload blocks, and mostly single load blocks.
        if (static_cast<float>(numSingle)/metrics.size() > 0.45
                && numMulti == 0)
        {
            return true;
        }
        return false;
    }

    // Check that ZMW "is sequencing" using base rate and width.
    const size_t numMetrics = metrics.size();
    uint32_t numFrames = 0;
    uint32_t numBaseFrames = 0;
    uint32_t numBases = 0;
    for (size_t nm = 0; nm < numMetrics; nm++)
    {
        numBaseFrames += metrics[nm].NumBaseFrames();
        numBases += metrics[nm].NumBases();
        numFrames += metrics[nm].TraceMetrics().NumFrames();
    }

    float baseRate = static_cast<float>(numBases / (numFrames / rtConfig_->frameRate));
    float baseWidth = numBases != 0 ? static_cast<float>((numBaseFrames / numBases) / rtConfig_->frameRate) : 0;

    bool lowerThresh = (rtConfig_->minBaseRate <= baseRate && rtConfig_->minBaseWidth <= baseWidth);
    bool upperThresh = (rtConfig_->maxBaseRate  < 0 || baseRate  < rtConfig_->maxBaseRate) && 
                       (rtConfig_->maxBaseWidth < 0 || baseWidth < rtConfig_->maxBaseWidth);

    return rtConfig_->newJsonFormat ? (lowerThresh && upperThresh) : lowerThresh;
}

template <typename TMetrics>
void ZmwMetricsRecorder<TMetrics>::ComputeMetrics()
{
    while (sendMetricsThreadContinue_)
    {
        std::unique_ptr<MetricsSuperChunkBuffer> metricsSuperChunkBuffer;

        // Prevent the buffer queue from growing beyond an upper bound.
        if (metricsSuperChunkBufferQueue_.Size() >= rtConfig_->maxQueueSize)
        {
            // Discard the front half of the queue.
            const auto n = metricsSuperChunkBufferQueue_.Size() / 2;
            PBLOG_WARN << "Real-time metrics buffer queue over limit. Discarding "
                       << metricsSuperChunkBufferQueue_.Size() - n
                       << " buffers.";
            while (metricsSuperChunkBufferQueue_.Size() > n)
            {
                // Queue is not empty. So no time-out needed.
                metricsSuperChunkBufferQueue_.Pop(metricsSuperChunkBuffer, std::chrono::milliseconds(100));

                // Update the last seen timestamp and delta.
                lastChunkTimeStampStart_ = metricsSuperChunkBuffer->TimeStampStart();
                lastChunkTimeStampDelta_ = metricsSuperChunkBuffer->TimeStampDelta();
            }
        }

        if (metricsSuperChunkBufferQueue_.Pop(metricsSuperChunkBuffer, std::chrono::milliseconds(3000)))
        {
            auto numMetrics = metricsSuperChunkBuffer->NumMetricsBlocksPerSlice();
            PBLOG_INFO << "Computing rt-metrics for super chunk index="
                       << metricsSuperChunkBuffer->SuperChunkIndex()
                       << " for number of metric blocks=" << numMetrics;

            MetricsSuperChunk msc(numMetrics,
                                  metricsSuperChunkBuffer->TimeStampStart(),
                                  metricsSuperChunkBuffer->TimeStampDelta(),
                                  regions_, zmwRTRegions_, rtConfig_.get(), token_);

            size_t len = metricsSuperChunkBuffer->size();
            for (size_t i = 0; i < len; ++i)
            {
                const auto& data = (*metricsSuperChunkBuffer)[i];

                for (uint32_t nM = 0; nM < numMetrics; nM++)
                {
                    PBLOG_TRACE << "Categorizing ZMWs for rt-metrics for metricBlock=" << nM;

                    msc.MetricsBlocks()[nM].
                            StartFrame(data.metrics[nM].TraceMetrics().StartFrame()).
                            NumFrames(data.metrics[nM].TraceMetrics().NumFrames());

                    if ((data.metrics[nM].TraceMetrics().StartFrame() / GetPrimaryConfig().cache.framesPerSuperchunk)
                        != metricsSuperChunkBuffer->SuperChunkIndex())
                    {
                        // Assume it belongs to previous super chunk.
                        msc.MetricsBlocks()[nM].
                                ComputeTimeStamps(metricsSuperChunkBuffer->SuperChunkIndex() - 1,
                                                  lastChunkTimeStampStart_, lastChunkTimeStampDelta_);
                    }
                    else
                    {
                        msc.MetricsBlocks()[nM].
                                ComputeTimeStamps(metricsSuperChunkBuffer->SuperChunkIndex(),
                                                  metricsSuperChunkBuffer->TimeStampStart(),
                                                  metricsSuperChunkBuffer->TimeStampDelta());
                    }

                    const auto& position = rtConfig_->unitCellTypes[data.zmwIndex].second;
                    bool isSequencingZmw = IsSequencingZmw(data.metrics);
                    msc.MetricsBlocks()[nM].AddZmw(data.metrics[nM], position, isSequencingZmw);
                }
            }

            // If a metrics file is specified, output to that file instead.
            if (metricsOut_.is_open())
            {
                metricsOut_ << msc.ToCSV();


                // Send the metrics.
                Json::FastWriter fastWriter;
                std::string text = fastWriter.write(msc.ToJson());
                PBLOG_INFO << text;
            }
            else
            {
                PBLOG_INFO << "Sending rt-metrics to PAWS for super chunk index="
                           << metricsSuperChunkBuffer->SuperChunkIndex();

                // Send the metrics.
                Json::FastWriter fastWriter;
                std::string text = fastWriter.write(msc.ToJson());
                PBLOG_TRACE << text;
                std::unique_ptr<PacBio::IPC::BigAnnouncement> msg(
                    new PacBio::IPC::BigAnnouncement("basewriter/rtmetrics", text));
                metricsPublisher_->Send(*msg);
            }

            // Update the last seen timestamp and delta.
            lastChunkTimeStampStart_ = metricsSuperChunkBuffer->TimeStampStart();
            lastChunkTimeStampDelta_ = metricsSuperChunkBuffer->TimeStampDelta();

            metricsSuperChunkBuffer->Reset();
            idleMetricsSuperChunkBufferQueue_.Push(std::move(metricsSuperChunkBuffer));
        }

    }
}

template <typename TMetrics>
ZmwMetricsRecorder<TMetrics>::~ZmwMetricsRecorder() noexcept
{
    try
    {
        sendMetricsThreadContinue_ = false;
        if (sendMetricsThread_.joinable()) sendMetricsThread_.join();
        if (metricsOut_.is_open()) metricsOut_.close();
    }
    catch(const std::exception& ex)
    {
        PBLOG_ERROR << "ZmwMetricsRecorder::~ZmwMetricsRecorder caught exception:" << ex.what();
    }
    catch(...)
    {
        std::cerr << "Uncaught exception caught in ~ZmwMetricsRecorder "  << PacBio::GetBackTrace(5);
        PBLOG_FATAL << "Uncaught exception caught in ~ZmwMetricsRecorder " << PacBio::GetBackTrace(5);
        PacBio::Logging::PBLogger::Flush();
        std::terminate();
    }
}

template <typename TMetrics>
void ZmwMetricsRecorder<TMetrics>::WaitForTermination()
{
    // wait for buffer to empty
    while (metricsSuperChunkBufferQueue_.Size() > 0)
    {
        PacBio::POSIX::Sleep(0.010);
    }

    sendMetricsThreadContinue_ = false;
    if (sendMetricsThread_.joinable()) sendMetricsThread_.join();
    if (metricsOut_.is_open()) metricsOut_.close();
}

// this can be deleted after we have more time sitting on this code change.
#define SPI1451_INSTRUMENTATION
#ifdef SPI1451_INSTRUMENTATION
struct
{
    double t0;
    uint32_t numSuperChunkBuffersAcquired;
    double totalAllocationTime;
} spi1451 = {0.0, 0, 0.0};
#endif

template <typename TMetrics>
bool ZmwMetricsRecorder<TMetrics>::FlushMetrics(uint32_t superChunkIndex, uint64_t timeStampStart, uint64_t timeStampDelta)
{
    PBLOG_INFO << "ZmwMetricsRecorder::FlushMetrics superchunk:"
                << currentMetricsSuperChunkBuffer_->SuperChunkIndex()
                << " timeStampStart=" << timeStampStart
                << " timeStampDelta=" << timeStampDelta
                << " empty=" << currentMetricsSuperChunkBuffer_->empty();

    currentMetricsSuperChunkBuffer_->SuperChunkIndex(superChunkIndex).
            TimeStampStart(timeStampStart).
            TimeStampDelta(timeStampDelta);

#ifdef SPI1451_INSTRUMENTATION
    double t = PacBio::Utilities::Time::GetMonotonicTime();
#endif
    if (!currentMetricsSuperChunkBuffer_->empty())
    {
        // Push the metrics onto the processing queue.
        metricsSuperChunkBufferQueue_.Push(std::move(currentMetricsSuperChunkBuffer_));

        currentMetricsSuperChunkBuffer_ = idleMetricsSuperChunkBufferQueue_.Pop();
    }

#ifdef SPI1451_INSTRUMENTATION
    double t1 = PacBio::Utilities::Time::GetMonotonicTime();
    spi1451.numSuperChunkBuffersAcquired++;
    double rate = 0;
    if (spi1451.t0 != 0.0)
    {
        double elapsedTime = t - spi1451.t0;
        rate = (spi1451.numSuperChunkBuffersAcquired -1) / elapsedTime;
    }
    else
    {
        spi1451.t0 = t;
    }

    double allocationTime = t1 - t;
    spi1451.totalAllocationTime += allocationTime;
    PBLOG_INFO << "SPI1451 numSuperChunkBuffersAcquired: " << spi1451.numSuperChunkBuffersAcquired <<
                " superChunkBufferRate:" << rate <<
                " allocationTime:" << allocationTime <<
                " totalAllocationTime:" << spi1451.totalAllocationTime;
#endif

    return true;
}

template <typename TMetrics>
bool ZmwMetricsRecorder<TMetrics>::AddZmwSlice(const TMetrics* hfMetrics, const uint32_t numMetrics,
                                               const uint32_t numBaseEvents, const uint32_t zmwId)
{
    if (numMetrics > 0)
    {
        // Check if we have overflowed the basecall buffer. If so, the ZMW is no longer used for real-time metrics.
        if (speedingZMWs.find(zmwId) == speedingZMWs.end() && numBaseEvents == maxSamples_)
        {
            PBLOG_WARN << "Zmw=" << zmwId << " overflowed basecall buffer, will no longer be used for real-time metrics";
            speedingZMWs.insert(zmwId);
        }
        else
        {
            assert(hfMetrics != nullptr);
            // Copy the set of metrics into the current buffer.
            currentMetricsSuperChunkBuffer_->AddMetricBlocks(zmwId, numMetrics, hfMetrics);
        }
    }
    else
    {
        if (currentMetricsSuperChunkBuffer_->SuperChunkIndex() == 0xFFFFFFFF)
        {
            if (warningsForThisSuperChunk_ < 100)
            {
                PBLOG_WARN << "currentMetricsSuperChunkBuffer does not have a SuperChunkIndex set";
            }
        }
        else
        {
            if (lastSuperChunkIndex_ != currentMetricsSuperChunkBuffer_->SuperChunkIndex())
            {
                warningsForThisSuperChunk_ = 0;
            }
        }
        lastSuperChunkIndex_ = currentMetricsSuperChunkBuffer_->SuperChunkIndex();
        warningsForThisSuperChunk_++;
        if (warningsForThisSuperChunk_ < 100)
        {
            PBLOG_DEBUG << "No metrics for zmwId=" << zmwId;
            if (warningsForThisSuperChunk_ >= 100)
            {
                PBLOG_DEBUG << "Too many empty ZMWs, will suppress warning message until next superchunk";
            }
        }
        else
        {
            PBLOG_TRACE << "No metrics for zmwId=" << zmwId;
        }
    }

    return true;
}


//
// Explicit Instantiation
//

template class ZmwMetricsRecorder<PacBio::Primary::BasecallingMetricsT::Sequel>;
template class ZmwMetricsRecorder<PacBio::Primary::BasecallingMetricsT::Spider>;

}}
