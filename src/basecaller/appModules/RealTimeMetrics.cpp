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
    Impl(uint32_t framesPerHFMetricBlock, size_t numBatches,
         std::vector<Data::RealTimeMetricsRegion>&& regions,
         std::vector<DataSourceBase::LaneSelector>&& selections,
         const std::vector<std::vector<uint32_t>>& zmwFeatures,
         float frameRate, const std::string& rtMetricsFile,
         bool useSingleActivityLabels)
    : framesPerHFMetricBlock_{framesPerHFMetricBlock}
    , numBatches_{numBatches}
    , frameRate_{frameRate}
    , rtMetricsFile_{rtMetricsFile}
    , useSingleActivityLabels_{useSingleActivityLabels}
    , jsonWriter_{GetStreamWriterBuilder().newStreamWriter()}
    {
        std::vector<std::string> regionNames;
        for (size_t i = 0; i < regions.size(); i++)
        {
            uint32_t featuresMask = std::accumulate(regions[i].featuresForFilter.begin(),
                                                    regions[i].featuresForFilter.end(), 0,
                                                    [](uint32_t a, uint32_t b) { return a | b; });
            regionNames.push_back(regions[i].name);
            regionInfo_.push_back({ std::move(regions[i]), std::move(selections[i]),
                                    SelectedLanesWithFeatures(zmwFeatures[i], featuresMask) });
        }

        if (rtMetricsFile_ != "")
        {
            rtMetricsOut_.open(rtMetricsFile_, std::ios_base::trunc);
        }
    }

    ~Impl() = default;

    Json::StreamWriterBuilder GetStreamWriterBuilder()
    {
        Json::StreamWriterBuilder builder;
        builder.settings_["commentStyle"] = "None";
        builder.settings_["indentation"] = "";
        return builder;
    }

    void Process(const Mongo::Data::BatchResult& in)
    {
        const auto& pulseBatch = in.pulses;
        const auto& metricsPtr = in.metrics;

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

        if (metricsPtr)
        {
            // Only record metrics when we get a full metric block worth, lanes in a pool
            // should have the same metric block size.
            if (metricsPtr->GetHostView()[0].numFrames[0] == framesPerHFMetricBlock_)
            {
                fullMetricsBatchesSeen_++;
                const auto zmwOffset = pulseBatch.GetMeta().FirstZmw();
                const DataSourceBase::LaneIndex laneBegin = zmwOffset / pulseBatch.Dims().laneWidth;
                const DataSourceBase::LaneIndex laneEnd = laneBegin + pulseBatch.Dims().lanesPerBatch;

                for (auto& r: regionInfo_)
                {
                    for (const auto laneIdx: r.selection.SelectedLanes(laneBegin, laneEnd))
                    {
                        auto& metrics = metricsPtr->GetHostView()[laneIdx];

                        // Need to convert the activity label to uint16_t as LaneArray doesn't support uint8_t.
                        Cuda::Utility::CudaArray<uint16_t, laneSize> aL;
                        std::transform(metrics.activityLabel.begin(), metrics.activityLabel.end(), aL.begin(),
                                       [](const Data::HQRFPhysicalStates& s) { return static_cast<uint16_t>(s); });
                        LaneArray<uint16_t> activityLabel(aL);

                        auto mask = r.laneMasks[laneIdx];

                        // Filter for zmws marked as SINGLE by the real-time activity labeler.
                        if (useSingleActivityLabels_)
                            mask &= (activityLabel == static_cast<uint16_t>(Data::HQRFPhysicalStates::SINGLE));

                        LaneArray<float> numBases(LaneArray<uint16_t>(metrics.numBases));
                        r.ma.baseRate.AddSample(numBases / framesPerHFMetricBlock_, mask);

                        LaneArray<float> numBaseFrames(LaneArray<uint16_t>(metrics.numBaseFrames));
                        r.ma.baseWidth.AddSample((numBaseFrames / numBases) / frameRate_, mask & (numBases != 0));

                        LaneArray<float> numPulses(LaneArray<uint16_t>(metrics.numPulses));
                        r.ma.pulseRate.AddSample(numPulses / framesPerHFMetricBlock_, mask);

                        LaneArray<float> numPulseFrames(LaneArray<uint16_t>(metrics.numPulseFrames));
                        r.ma.pulseWidth.AddSample((numPulseFrames / numPulses) / frameRate_, mask & (numPulses != 0));

                        LaneArray<float> baseline(metrics.frameBaselineDWS);
                        r.ma.baseline.AddSample(baseline, mask);

                        LaneArray<float> baselineVar(metrics.frameBaselineVarianceDWS);
                        auto baselineSd = sqrt(baselineVar);
                        r.ma.baselineSd.AddSample(baselineSd, mask & (baselineSd != 0));

                        for (size_t i = 0; i < numAnalogs; i++)
                        {
                            LaneArray<float> pkmid(metrics.pkMidSignal[i]);
                            LaneArray<float> pkmidFrames(LaneArray<uint16_t>(metrics.numPkMidFrames[i]));
                            const auto pkmidMean = pkmid / pkmidFrames;

                            r.ma.pkmid[i].AddSample(pkmidMean, mask & (pkmidFrames != 0));
                            r.ma.snr[i].AddSample(pkmidMean / baselineSd, mask & (baselineSd != 0));
                        }
                    }
                }
            }
        }

        // Require a full chip worth of full metrics.
        if (fullMetricsBatchesSeen_ == numBatches_)
        {
            Json::Value rtMetrics;
            for (auto& r: regionInfo_)
            {
                Data::RealTimeMetricsReport report;
                report.name = r.region.name;
                report.startFrame = currFrame_;
                report.numFrames = framesPerHFMetricBlock_;

                // TODO
                // report.beginFrameTimeStamp =
                // report.endFrameTimeStamp =

                r.FillReportMetrics(report);
                rtMetrics.append(report.Serialize());
                r.ma = RegionInfo::MetricAccumulators();
            }

            // TODO: Emit report.

            if (rtMetricsOut_.is_open())
            {
                jsonWriter_->write(rtMetrics, &rtMetricsOut_);
                rtMetricsOut_ << std::endl;
            }

            fullMetricsBatchesSeen_ = 0;
        }

        batchesSeen_++;
        if (batchesSeen_ == numBatches_)
        {
            for (auto& r : regionInfo_)
            {
                r.ma = RegionInfo::MetricAccumulators();
            }
            fullMetricsBatchesSeen_ = 0;
            batchesSeen_ = 0;
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
            // Count total number of zmws for this region.
            for (const auto& lm : laneMasks)
            {
                for (size_t i = 0; i < laneSize; i++)
                    if (lm[i]) totalZmws++;
            }
        }

        Mongo::Data::RealTimeMetricsRegion region;
        DataSource::DataSourceBase::LaneSelector selection;
        std::vector<Mongo::LaneMask<>> laneMasks;

        using FloatArray = Mongo::LaneArray<float>;
        using MetricAccumulator = Mongo::StatAccumulator<FloatArray>;
        using AnalogMetricAccumulator = std::array<MetricAccumulator,Mongo::numAnalogs>;

        struct MetricAccumulators
        {
            MetricAccumulator baseRate;
            MetricAccumulator baseWidth;
            MetricAccumulator pulseRate;
            MetricAccumulator pulseWidth;
            AnalogMetricAccumulator snr;
            AnalogMetricAccumulator pkmid;
            MetricAccumulator baseline;
            MetricAccumulator baselineSd;
        };
        MetricAccumulators ma;
        uint32_t totalZmws = 0;

        void FillReportMetrics(Mongo::Data::RealTimeMetricsReport& report)
        {
            FillSummaryStats(ma.baseRate, report.baseRate);
            FillSummaryStats(ma.baseWidth, report.baseWidth);
            FillSummaryStats(ma.pulseRate, report.pulseRate);
            FillSummaryStats(ma.pulseWidth, report.pulseWidth);

            for (size_t i = 0; i < Mongo::numAnalogs; i++)
            {
                FillSummaryStats(ma.snr[i], report.snr[i]);
                FillSummaryStats(ma.pkmid[i], report.pkmid[i]);
            }

            FillSummaryStats(ma.baseline, report.baseline);
            FillSummaryStats(ma.baselineSd, report.baselineSd);
        }

        void FillSummaryStats(const MetricAccumulator& ma, Mongo::Data::RealTimeMetricsReport::SummaryStats& stats)
        {
            stats.sampleTotal = totalZmws;

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
                stats.sampleMean = summedM1 / stats.sampleSize;
                float sampleVar = (summedM2 - ((summedM1 * summedM1) / stats.sampleSize) / (stats.sampleSize - 1));
                stats.sampleCV = sqrt(sampleVar) / stats.sampleMean;

                // TODO: Need to compute the median. Previously we
                // were using boost::accumulators which uses a p^2
                // quantile estimator.
            }
        }
    };
    std::vector<RegionInfo> regionInfo_;

    uint32_t framesPerHFMetricBlock_;
    size_t numBatches_;
    float frameRate_;
    std::string rtMetricsFile_;
    bool useSingleActivityLabels_;
    std::ofstream rtMetricsOut_;
    std::unique_ptr<Json::StreamWriter> jsonWriter_;

    size_t batchesSeen_ = 0;
    size_t fullMetricsBatchesSeen_ = 0;
    int32_t currFrame_ = std::numeric_limits<int32_t>::min();
};

RealTimeMetrics::RealTimeMetrics(uint32_t framesPerMetricBlock, size_t numBatches,
                                 std::vector<Data::RealTimeMetricsRegion>&& regions,
                                 std::vector<DataSourceBase::LaneSelector>&& selections,
                                 const std::vector<std::vector<uint32_t>>& features,
                                 float frameRate, const std::string& csvOutputFile,
                                 bool useSingleActivityLabels)
    : impl_(std::make_unique<Impl>(framesPerMetricBlock, numBatches, std::move(regions),
                                   std::move(selections), features, frameRate,
                                   csvOutputFile, useSingleActivityLabels))
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
