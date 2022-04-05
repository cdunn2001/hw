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

#include <pacbio/logging/Logger.h>

#include <appModules/RealTimeMetrics.h>

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;

namespace PacBio::Application
{

class RealTimeMetrics::Impl
{
public:
    Impl(uint32_t framesPerMetricBlock, size_t numBatches,
         std::vector<Data::RealTimeMetricsRegion>&& regions,
         std::vector<DataSourceBase::LaneSelector>&& selections,
         const std::vector<std::vector<uint32_t>>& features,
         float frameRate, const std::string& csvOutputFile)
    : framesPerMetricBlock_{framesPerMetricBlock}
    , numBatches_{numBatches}
    , frameRate_{frameRate}
    , csvOutputFile_{csvOutputFile}
    {
        std::vector<std::string> regionNames;
        for (size_t i = 0; i < regions.size(); i++)
        {
            uint32_t featuresMask = static_cast<uint32_t>(regions[i].features.front());
            for (size_t f = 1; f < regions[i].features.size(); f++)
                featuresMask |= static_cast<uint32_t>(regions[i].features[f]);
            regionInfo_.push_back({ std::move(regions[i]), std::move(selections[i]),
                                    SelectedLanesWithFeatures(features[i], featuresMask) });
            regionNames.push_back(regions[i].name);
        }

        if (csvOutputFile_ != "")
        {
            rtMetricsCsvOut_.open(csvOutputFile_, std::ios_base::trunc);
            rtMetricsCsvOut_ << Mongo::Data::RealTimeMetricsReport::Header(regionNames) << std::endl;
        }
    }

    ~Impl()
    {
        if (rtMetricsCsvOut_.is_open()) rtMetricsCsvOut_.close();
    }

    void Process(const Mongo::Data::BatchResult& in)
    {
        const auto& pulseBatch = in.pulses;
        const auto& metricsPtr = in.metrics;

        if (metricsPtr)
        {
            if (metricsPtr->GetHostView()[0].numFrames[0] == framesPerMetricBlock_)
            {
                if (pulseBatch.GetMeta().FirstFrame() > currFrame_)
                {
                    if (batchesSeen_ % numBatches_ != 0)
                        throw PBException("Data out of order, new metric block seen before all batches of previous metric block");
                    currFrame_ = pulseBatch.GetMeta().FirstFrame();
                }
                else if (pulseBatch.GetMeta().FirstFrame() < currFrame_)
                {
                    throw PBException("Data out of order, multiple metric blocks being processed simultaneously");
                }

                batchesSeen_++;

                const auto zmwOffset = pulseBatch.GetMeta().FirstZmw();
                const DataSourceBase::LaneIndex laneBegin = zmwOffset / pulseBatch.Dims().laneWidth;
                const DataSourceBase::LaneIndex laneEnd = laneBegin + pulseBatch.Dims().lanesPerBatch;

                for (auto& r : regionInfo_)
                {
                    for (const auto laneIdx : r.selection.SelectedLanes(laneBegin, laneEnd))
                    {
                        auto& metrics = metricsPtr->GetHostView()[laneIdx];

                        // Need to convert the activity label to uint16_t as LaneArray doesn't support uint8_t.
                        Cuda::Utility::CudaArray<uint16_t, laneSize> aL;
                        std::transform(metrics.activityLabel.begin(), metrics.activityLabel.end(), aL.begin(),
                                       [](const Data::HQRFPhysicalStates& s) { return static_cast<uint16_t>(s); });
                        LaneArray<uint16_t> activityLabel(aL);

                        // Filter for zmws marked as SINGLE by the real-time activity labeler.
                        const auto& mask = r.laneMasks[laneIdx] &
                                           (activityLabel == static_cast<uint16_t>(Data::HQRFPhysicalStates::SINGLE));

                        LaneArray<float> numBases(LaneArray<uint16_t>(metrics.numBases));
                        r.baseRate.AddSample(numBases / framesPerMetricBlock_, mask);

                        LaneArray<float> numBaseFrames(LaneArray<uint16_t>(metrics.numBaseFrames));
                        r.baseWidth.AddSample((numBaseFrames / numBases) / frameRate_, mask);

                        LaneArray<float> numPulses(LaneArray<uint16_t>(metrics.numPulses));
                        r.pulseRate.AddSample(numPulses / framesPerMetricBlock_, mask);

                        LaneArray<float> numPulseFrames(LaneArray<uint16_t>(metrics.numPulseFrames));
                        r.pulseWidth.AddSample((numPulseFrames / numPulses) / frameRate_, mask);

                        LaneArray<float> baseline(metrics.frameBaselineDWS);
                        r.baseline.AddSample(baseline, mask);

                        LaneArray<float> baselineVar(metrics.frameBaselineVarianceDWS);
                        r.baselineSd.AddSample(sqrt(baselineVar), mask);

                        for (size_t i = 0; i < numAnalogs; i++)
                        {
                            LaneArray<float> pkmid(metrics.pkMidSignal[i]);
                            LaneArray<float> pkmidFrames(LaneArray<uint16_t>(metrics.numPkMidFrames[i]));
                            const auto pkmidMean = pkmid / pkmidFrames;

                            r.pkmid[i].AddSample(pkmidMean, mask);
                            r.snr[i].AddSample(pkmidMean / sqrt(baselineVar), mask);
                        }
                    }
                }

                if (batchesSeen_ == numBatches_)
                {
                    if (rtMetricsCsvOut_.is_open())
                    {
                        // TODO
                        float beginFrameTimeStamp = 0;
                        float endFrameTimeStamp = 0;

                        rtMetricsCsvOut_ << currFrame_ << ","
                                         << framesPerMetricBlock_ << ","
                                         << beginFrameTimeStamp << ","
                                         << endFrameTimeStamp << ",";
                    }

                    Json::Value rtMetrics;
                    for (auto& r : regionInfo_)
                    {
                        Data::RealTimeMetricsReport report;
                        report.name = r.region.name;
                        report.startFrame = currFrame_;
                        report.numFrames = framesPerMetricBlock_;

                        // TODO
                        // report.beginFrameTimeStamp =
                        // report.endFrameTimeStamp =

                        r.FillReportMetrics(report);
                        rtMetrics.append(report.Serialize());
                        if (rtMetricsCsvOut_.is_open())
                        {
                            rtMetricsCsvOut_ << report.CSVOutput();
                        }

                        r.ResetAccumulators();
                    }

                    // Emit report.

                    if (rtMetricsCsvOut_.is_open())
                    {
                        rtMetricsCsvOut_ << std::endl;
                    }

                    batchesSeen_ = 0;
                }
            }
        }
    }

private:
    struct RegionInfo
    {
        RegionInfo(Mongo::Data::RealTimeMetricsRegion&& r,
                   DataSource::DataSourceBase::LaneSelector&& s,
                   std::vector<Mongo::LaneMask<>>&& l)
            : region(std::move(r))
            , selection(std::move(s))
            , laneMasks(std::move(l))
        {
            for (const auto& lm : laneMasks)
            {
                for (size_t i = 0; i < laneSize; i++)
                    if (lm[i]) total++;
            }
        }

        void ResetAccumulators()
        {
            baseRate.Reset();
            baseWidth.Reset();
            pulseRate.Reset();
            pulseWidth.Reset();
            for (size_t i = 0; i < numAnalogs; i++)
            {
                snr[i].Reset();
                pkmid[i].Reset();
            }
            baseline.Reset();
            baselineSd.Reset();
        }

        Mongo::Data::RealTimeMetricsRegion region;
        DataSource::DataSourceBase::LaneSelector selection;
        std::vector<Mongo::LaneMask<>> laneMasks;

        using FloatArray = Mongo::LaneArray<float>;
        using MetricAccumulator = Mongo::StatAccumulator<FloatArray>;
        using AnalogMetricAccumulator = std::array<MetricAccumulator,Mongo::numAnalogs>;

        MetricAccumulator baseRate;
        MetricAccumulator baseWidth;
        MetricAccumulator pulseRate;
        MetricAccumulator pulseWidth;
        AnalogMetricAccumulator snr;
        AnalogMetricAccumulator pkmid;
        MetricAccumulator baseline;
        MetricAccumulator baselineSd;
        uint32_t total = 0;

        void FillReportMetrics(Mongo::Data::RealTimeMetricsReport& report)
        {
            FillSummaryStats(baseRate, report.baseRate);
            FillSummaryStats(baseWidth, report.baseWidth);
            FillSummaryStats(pulseRate, report.pulseRate);
            FillSummaryStats(pulseWidth, report.pulseWidth);

            for (size_t i = 0; i < Mongo::numAnalogs; i++)
            {
                FillSummaryStats(snr[i], report.snr[i]);
                FillSummaryStats(pkmid[i], report.pkmid[i]);
            }

            FillSummaryStats(baseline, report.baseline);
            FillSummaryStats(baselineSd, report.baselineSd);
        }

        void FillSummaryStats(const MetricAccumulator& ma, Mongo::Data::RealTimeMetricsReport::SummaryStats& stats)
        {
            stats.sampleTotal = total;

            const auto counts = MakeUnion(ma.Count());
            const auto m1 = MakeUnion(ma.M1());
            const auto m2 = MakeUnion(ma.M2());
            float summedM1 = 0;
            float summedM2 = 0;
            for (size_t i = 0; i < laneSize; i++)
            {
                stats.sampleSize += counts[i];
                summedM1 += m1[0];
                summedM2 += m2[0];
            }

            if (stats.sampleSize > region.minSampleSize)
            {
                if (stats.sampleSize > 0)
                    stats.sampleMean = summedM1 / stats.sampleSize;
                if (stats.sampleSize >= 2)
                {
                    float sampleVar = (summedM2 - ((summedM1 * summedM1) / stats.sampleSize) / (stats.sampleSize - 1));
                    stats.sampleCV = sqrt(sampleVar) / stats.sampleMean;
                }

                // TODO: median?
            }
        }
    };
    std::vector<RegionInfo> regionInfo_;

    uint32_t framesPerMetricBlock_;
    size_t numBatches_;
    float frameRate_;
    std::string csvOutputFile_;
    std::ofstream rtMetricsCsvOut_;

    size_t batchesSeen_ = 0;
    int32_t currFrame_ = std::numeric_limits<int32_t>::min();
};

RealTimeMetrics::RealTimeMetrics(uint32_t framesPerMetricBlock, size_t numBatches,
                                 std::vector<Data::RealTimeMetricsRegion>&& regions,
                                 std::vector<DataSourceBase::LaneSelector>&& selections,
                                 const std::vector<std::vector<uint32_t>>& features,
                                 float frameRate, const std::string& csvOutputFile)
    : impl_(std::make_unique<Impl>(framesPerMetricBlock, numBatches, std::move(regions),
                                   std::move(selections), features, frameRate,
                                   csvOutputFile))
{ }

RealTimeMetrics::~RealTimeMetrics() = default;

void RealTimeMetrics::Process(const Mongo::Data::BatchResult& in)
{
    impl_->Process(in);
}

std::vector<LaneMask<>> RealTimeMetrics::SelectedLanesWithFeatures(const std::vector<uint32_t>& features,
                                                                   uint32_t featuresMask)
{
    LaneArray<uint32_t> fm {featuresMask};
    std::vector<LaneMask<>> laneMasks;

    assert(features.size() % laneSize == 0);
    for (size_t i = 0; i < features.size(); i += laneSize)
    {
        LaneArray<uint32_t> lf(MemoryRange<uint32_t,laneSize>(features.data()+i));
        laneMasks.emplace_back((fm & LaneArray<uint32_t>(lf)) == fm);
    }

    return laneMasks;
}

} // namespace PacBio::Application
