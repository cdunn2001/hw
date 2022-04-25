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

#include <boost/filesystem.hpp>

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
         const Data::RealTimeMetricsConfig& rtConfig,
         std::vector<DataSourceBase::LaneSelector>&& selections,
         const std::vector<std::vector<uint32_t>>& zmwFeatures,
         float frameRate)
    : framesPerHFMetricBlock_{framesPerHFMetricBlock}
    , numBatches_{numBatches}
    , frameRate_{frameRate}
    , jsonFileName_{rtConfig.jsonOutputFile}
    , csvFileName_{rtConfig.csvOutputFile}
    , useSingleActivityLabels_{rtConfig.useSingleActivityLabels}
    , jsonWriter_{GetStreamWriterBuilder().newStreamWriter()}
    {
        std::vector<std::string> regionNames;
        for (size_t i = 0; i < rtConfig.regions.size(); i++)
        {
            uint32_t featuresMask = std::accumulate(rtConfig.regions[i].featuresForFilter.begin(),
                                                    rtConfig.regions[i].featuresForFilter.end(), 0,
                                                    [](uint32_t a, uint32_t b) { return a | b; });
            regionNames.push_back(rtConfig.regions[i].name);
            regionInfo_.push_back({rtConfig.regions[i], std::move(selections[i]),
                                    SelectedLanesWithFeatures(zmwFeatures[i], featuresMask) });
        }

        if (csvFileName_ != "" && boost::filesystem::exists(csvFileName_))
        {
            PBLOG_INFO << "RTMetrics file " << csvFileName_ << " already exists, attempting to remove...";
            if (boost::filesystem::is_regular_file(csvFileName_))
            {
                boost::filesystem::remove(csvFileName_);
            } else
            {
                throw PBException("RTMetrics file " + csvFileName_ +
                                  " is not a regular file that can be removed");
            }
            PBLOG_INFO << "RTMetrics file " << csvFileName_ << " removed";
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
                        auto& metrics = metricsPtr->GetHostView()[laneIdx - laneBegin];

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
            Data::RealTimeMetricsReport report;
            // TODO PTSD-1513
            //report.frameTimeStampDelta =
            //report.startFrameTimeStamp =

            report.metricsChunk.numMetricsBlocks = 1;
            report.metricsChunk.metricsBlocks.resize(1);

            auto& blockReport = report.metricsChunk.metricsBlocks.front();
            // TODO PTSD-1513
            //blockReport.beginFrameTimeStamp =
            //blockReport.endFrameTimeStamp =
            blockReport.numFrames = framesPerHFMetricBlock_;
            blockReport.startFrame = currFrame_;
            blockReport.groups.reserve(regionInfo_.size());
            for (auto& r: regionInfo_)
            {
                auto& regionReport = blockReport.groups.emplace_back();
                regionReport.region = r.region.name;

                r.FillReportMetrics(regionReport);
                r.ma = RegionInfo::MetricAccumulators();
            }


            if (csvFileName_ != "")
            {
                if (!rtMetricsCsvOut_.is_open())
                {
                    rtMetricsCsvOut_.open(csvFileName_, std::ios_base::trunc);
                    assert(!report.metricsChunk.metricsBlocks.empty());
                    rtMetricsCsvOut_ << report.metricsChunk.metricsBlocks[0].GenerateCSVHeader();
                }
                for (const auto& block : report.metricsChunk.metricsBlocks)
                {
                    rtMetricsCsvOut_ << block.GenerateCSVRow();
                }
            }

            if (jsonFileName_ != "")
            {
                std::string tmpName = jsonFileName_ + ".tmp";
                std::ofstream rtMetricsOut(tmpName, std::ios_base::trunc);
                jsonWriter_->write(report.Serialize(), &rtMetricsOut);
                rtMetricsOut << std::endl;

                rtMetricsOut.close();
                boost::filesystem::rename(tmpName, jsonFileName_);
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
        RegionInfo(const Mongo::Data::RealTimeMetricsRegion& r,
                   DataSource::DataSourceBase::LaneSelector&& s,
                   std::vector<Mongo::LaneMask<>>&& l)
            : region(r)
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

        void FillReportMetrics(Mongo::Data::GroupStats& groupReport)
        {
            for (auto metric : this->region.metrics)
            {
                auto& metricReport = groupReport.metrics.emplace_back();
                metricReport.name = metric;
                switch(metric)
                {
                    case Mongo::Data::MetricNames::Baseline:
                    {
                        FillSummaryStats(ma.baseline, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::BaselineStd:
                    {
                        FillSummaryStats(ma.baselineSd, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::Pkmid:
                    {
                        FillSummaryStats(ma.pkmid, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::SNR:
                    {
                        FillSummaryStats(ma.snr, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::PulseRate:
                    {
                        FillSummaryStats(ma.pulseRate, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::PulseWidth:
                    {
                        FillSummaryStats(ma.pulseWidth, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::BaseRate:
                    {
                        FillSummaryStats(ma.baseRate, metricReport);
                        break;
                    }
                    case Mongo::Data::MetricNames::BaseWidth:
                    {
                        FillSummaryStats(ma.baseWidth, metricReport);
                        break;
                    }
                }
            }
        }

        void FillSummaryStats(const MetricAccumulator& ma, Mongo::Data::SummaryStats& stats)
        {
            const auto counts = MakeUnion(ma.Count());
            const auto m1 = MakeUnion(ma.M1());
            const auto m2 = MakeUnion(ma.M2());
            float summedM0 = 0;
            float summedM1 = 0;
            float summedM2 = 0;
            for (size_t i = 0; i < laneSize; i++)
            {
                summedM0 += counts[i];
                summedM1 += m1[i];
                summedM2 += m2[i];
            }

            stats.sampleTotal.push_back(totalZmws);
            stats.sampleSize.push_back(summedM0);

            if (summedM0 > region.minSampleSize)
            {
                float sampleMean = summedM1 / summedM0;
                float sampleVar = (summedM2 - ((summedM1 * summedM1) / summedM0) / (summedM0 - 1));
                stats.sampleMean.push_back(sampleMean);
                stats.sampleCV.push_back(sqrt(sampleVar) / sampleMean);

            } else
            {
                stats.sampleMean.push_back(-1);
                stats.sampleCV.push_back(-1);
            }
            // TODO: Need to compute the median. Previously we
            // were using boost::accumulators which uses a p^2
            // quantile estimator.
            stats.sampleMed.push_back(-1);
        }

        void FillSummaryStats(const AnalogMetricAccumulator& ma, Mongo::Data::SummaryStats& stats)
        {
            for (size_t i = 0; i < numAnalogs; ++i)
            {
                FillSummaryStats(ma[i], stats);
            }
        }
    };
    std::vector<RegionInfo> regionInfo_;

    uint32_t framesPerHFMetricBlock_;
    size_t numBatches_;
    float frameRate_;
    std::string jsonFileName_;
    std::string csvFileName_;
    bool useSingleActivityLabels_;
    std::ofstream rtMetricsCsvOut_;
    std::unique_ptr<Json::StreamWriter> jsonWriter_;

    size_t batchesSeen_ = 0;
    size_t fullMetricsBatchesSeen_ = 0;
    int32_t currFrame_ = std::numeric_limits<int32_t>::min();
};

RealTimeMetrics::RealTimeMetrics(uint32_t framesPerMetricBlock, size_t numBatches,
                                 const Data::RealTimeMetricsConfig& rtConfig,
                                 std::vector<DataSourceBase::LaneSelector>&& selections,
                                 const std::vector<std::vector<uint32_t>>& features,
                                 float frameRate)
    : impl_(std::make_unique<Impl>(framesPerMetricBlock, numBatches, rtConfig,
                                   std::move(selections), features, frameRate))
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
