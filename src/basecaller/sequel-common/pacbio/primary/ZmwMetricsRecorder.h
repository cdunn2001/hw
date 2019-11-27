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
/// \brief declarations for a class for recording metrics per ZMW


#pragma once

#include <iostream>
#include <cstring>
#include <set>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/variant.hpp>

#include <json/json.h>

#include <pacbio/ipc/MessageQueue.h>
#include <pacbio/ipc/ThreadSafeQueue.h>
#include <pacbio/smrtdata/Basecall.h>

#include <pacbio/primary/Acquisition_Setup.h>
#include <pacbio/primary/BasecallingMetrics.h>
#include <pacbio/primary/ChipLayout.h>
#include <pacbio/primary/MemoryBuffer.h>
#include <pacbio/primary/RTMetricsConfig.h>
#include <pacbio/primary/SequelDefinitions.h>
#include <pacbio/primary/TrancheTitle.h>
#include <pacbio/primary/UnitCell.h>

namespace PacBio {
namespace Primary {

class IZmwMetricsRecorder
{
public:
    virtual bool FlushMetrics(uint32_t superChunkIndex, uint64_t timeStampStart, uint64_t timeStampDelta) = 0;
    virtual void WaitForTermination() = 0;
    virtual ~IZmwMetricsRecorder() noexcept = default;
};

template <typename TMetrics>
class ZmwMetricsRecorder : public IZmwMetricsRecorder
{
public:
    class ZmwMetricsRecorderConfig;
public:
    static constexpr size_t NCam = TMetrics::NFilter;
    static constexpr size_t NAnalogs = 4;
public:
    class ContinuousDist
    {
    public:
        typedef boost::accumulators::accumulator_set<float,
            boost::accumulators::stats<
                boost::accumulators::tag::count,
                boost::accumulators::tag::mean,
                boost::accumulators::tag::median,
                boost::accumulators::tag::variance
            >> dist;
    public:
        dist& Values()
        { return values_; }

        float Mean() const
        { return boost::accumulators::mean(values_); }

        float Median() const
        { return boost::accumulators::median(values_); }

        size_t Size() const
        { return boost::accumulators::count(values_); }

        float Variance() const
        { return boost::accumulators::variance(values_); }

        float StdDev() const
        { return std::sqrt(Variance()); }

        Json::Value ToJson(uint32_t minSampleSize) const
        {
            Json::Value root;

            root["sampleSize"] = Size();
            root["sampleMean"] = -1;
            root["sampleMed"] = -1;
            root["sampleCV"] = -1;

            if (Size() > minSampleSize)
            {
                root["sampleSize"] = Size();
                root["sampleMean"] = Mean();
                root["sampleMed"] = Median();
                root["sampleCV"] = Mean() > 0 ? StdDev() / Mean() : -1;
            }

            return root;
        }

        std::string ToCSV(uint32_t minSampleSize) const
        {
            Json::Value v = ToJson(minSampleSize);

            std::ostringstream csvOutput;
            csvOutput << v["sampleMean"].asString() << ","
                      << v["sampleMed"].asString() << ","
                      << v["sampleCV"].asString() << ","
                      << v["sampleSize"].asString();

            return csvOutput.str();
        }

    private:
        dist values_;
    }; // ContinuousDist

    class ZmwRTRegionNew
    {
    public:
        // A raw pointer to the chip layout is used as the caller
        // is responsible for managing the chip layout object.
        ZmwRTRegionNew(const RTMetricsRegionNew& region, const ChipLayout* chipLayout)
        : chipLayout_(chipLayout)
        , name_(region.name)
        , xMin_(region.xMin)
        , xExtent_(region.xExtent)
        , yMin_(region.yMin)
        , yExtent_(region.yExtent)
        , minSampleSize_(region.minSampleSize)
        , samplingFactorStride_(region.samplingFactorStride)
        {
            for (const auto& f : region.zmwTypesFilter)
            {
                switch (f())
                {
                    case RTMetricsRegionNew::zmwType::SEQUENCING:
                        zmwFilters_.push_back(&ChipLayout::IsSequencing);
                        break;
                    case RTMetricsRegionNew::zmwType::PORSEQUENCING:
                        zmwFilters_.push_back(&ChipLayout::IsPorSequencing);
                        break;
                    case RTMetricsRegionNew::zmwType::LASERSCATTER:
                        zmwFilters_.push_back(&ChipLayout::IsLaserScatter);
                        break;
                    case RTMetricsRegionNew::zmwType::LPTITRATION0P0X:
                        zmwFilters_.push_back(&ChipLayout::IsLaserPower0p0x);
                        break;
                    case RTMetricsRegionNew::zmwType::LPTITRATION0P5X:
                        zmwFilters_.push_back(&ChipLayout::IsLaserPower0p5x);
                        break;
                    case RTMetricsRegionNew::zmwType::LPTITRATION1P5X:
                        zmwFilters_.push_back(&ChipLayout::IsLaserPower1p5x);
                        break;
                    case RTMetricsRegionNew::zmwType::LPTITRATION2P0X:
                        zmwFilters_.push_back(&ChipLayout::IsLaserPower2p0x);
                        break;
                    default:
                        throw PBException("Unknown ZMW filter type specified!");
                }
            }

            for (const auto& m : region.zmwMetricsReported)
            {
                reportedMetrics.push_back(m().toString());
            }
        }

        bool ContainsZMW(uint16_t holeX, uint16_t holeY) const
        {
            return UCInRegion(holeX, holeY) && ZmwFilters(holeX, holeY);
        }

        const std::string& Name() const
        { return name_; }

        const std::vector<std::string>& MetricsToReport() const
        { return reportedMetrics; }

        uint32_t SamplingFactorStride() const
        { return samplingFactorStride_; }

        uint32_t MinSampleSize() const
        { return minSampleSize_; }

    private:
        bool ZmwFilters(uint16_t holeX, uint16_t holeY) const
        {
            for (const auto& filter: zmwFilters_)
            {
                if (!(this->chipLayout_->*filter)(holeX, holeY))
                    return false;
            }
            return true;
        }

        bool UCInRegion(uint16_t holeX, uint16_t holeY) const
        {
            return (xMin_ <= holeX && holeX < xMin_ + xExtent_) &&
                   (yMin_ <= holeY && holeY < yMin_ + yExtent_);
        }

        const ChipLayout* chipLayout_;
        std::string name_;
        uint32_t xMin_;
        uint32_t xExtent_;
        uint32_t yMin_;
        uint32_t yExtent_;
        uint32_t minSampleSize_;
        uint32_t samplingFactorStride_;
        std::vector<bool (PacBio::Primary::ChipLayout::*)(uint16_t, uint16_t) const> zmwFilters_;
        std::vector<std::string> reportedMetrics;
    };

    class ZmwGroupingNew
    {
    public:
        using SingleMetric = std::array<ContinuousDist, 1>;
        using AnalogsMetric = std::array<ContinuousDist, NAnalogs>;
        using ChannelsMetric = std::array<ContinuousDist, NCam>;

        // The constructor takes raw pointers as the caller is responsible
        // for managing both of these objects.
        ZmwGroupingNew(const ZmwRTRegionNew* rtRegion, const ZmwMetricsRecorderConfig* rtConfig)
        : rtRegion_(rtRegion)
        , rtConfig_(rtConfig)
        , numZmws_(0)
        , totalZmws_(0)
        {}

        bool ContainsZMW(uint16_t holeX, uint16_t holeY) const
        {
            return rtRegion_->ContainsZMW(holeX, holeY);
        }

        const std::string& Name() const
        { return rtRegion_->Name(); }

        bool AddZmw(const PacBio::Primary::UnitCell& position,
                    const TMetrics& bm, bool isSequencingZmw);

        Json::Value ToJson() const;

        std::string ToCSV() const;

    private:
        template<size_t N>
        Json::Value ArrayToJson(const std::array<ContinuousDist, N>& arr) const
        {
            Json::Value root;
            Json::Value sampleTotal(Json::arrayValue);
            Json::Value sampleSizes(Json::arrayValue);
            Json::Value sampleMeans(Json::arrayValue);
            Json::Value sampleMedians(Json::arrayValue);
            Json::Value sampleCV(Json::arrayValue);

            for (size_t n = 0; n < N; n++)
            {
                sampleTotal.append(totalZmws_);
                sampleSizes.append(arr[n].Size());

                if (arr[n].Size() > rtRegion_->MinSampleSize())
                {
                    sampleMeans.append(arr[n].Mean());
                    sampleMedians.append(arr[n].Median());
                    sampleCV.append(arr[n].Mean() > 0 ? arr[n].StdDev() / arr[n].Mean() : -1);
                }
                else
                {
                    sampleMeans.append(-1);
                    sampleMedians.append(-1);
                    sampleCV.append(-1);
                }
            }

            root["sampleTotal"] = sampleTotal;
            root["sampleSize"] = sampleSizes;
            root["sampleMean"] = sampleMeans;
            root["sampleMed"] = sampleMedians;
            root["sampleCV"] = sampleCV;

            return root;
        }

        template<size_t N>
        std::string ArrayToCSV(const std::array<ContinuousDist, N>& arr) const
        {
            std::ostringstream csvOutput;

            Json::Value v = ArrayToJson(arr);

            for (unsigned int n = 0; n < N; n++)
            {
                csvOutput << v["sampleMean"][n].asString() << ","
                          << v["sampleMed"][n].asString() << ","
                          << v["sampleCV"][n].asString() << ","
                          << v["sampleSize"][n].asString() << ","
                          << v["sampleTotal"][n].asString();
                if (n + 1 < N) csvOutput << ",";
            }

            return csvOutput.str();
        }

        const ZmwRTRegionNew* rtRegion_;
        const ZmwMetricsRecorderConfig* rtConfig_;
        uint32_t       numZmws_;
        uint32_t       totalZmws_;
        SingleMetric   baseRate_;
        SingleMetric   baseWidth_;
        SingleMetric   pulseRate_;
        SingleMetric   pulseWidth_;
        AnalogsMetric  snr_;
        AnalogsMetric  pkmid_;
        ChannelsMetric baseline_;
        ChannelsMetric baselineStdDev_;
    };

    class ZmwGrouping
    {
    public:
        using AnalogsMetric = std::array<ContinuousDist, NAnalogs>;
        using ChannelsMetric = std::array<ContinuousDist, NCam>;

        ZmwGrouping() = default;

        ZmwGrouping(const std::string& name, const ZmwMetricsRecorderConfig* rtConfig,
                    bool (PacBio::Primary::ChipLayout::*filter)(uint16_t, uint16_t) const)
        : ZmwGrouping(name, rtConfig, filter, 0, 0xFFFFFF, 0, 0xFFFFFF)
        {}

        ZmwGrouping(const std::string& name, const ZmwMetricsRecorderConfig* rtConfig,
                    bool (PacBio::Primary::ChipLayout::*filter)(uint16_t, uint16_t) const,
                    uint32_t xMin, uint32_t xExtent, uint32_t yMin, uint32_t yExtent)
        : name_(name)
        , rtConfig_(rtConfig)
        , filter_(filter)
        , xMin_(xMin), xExtent_(xExtent)
        , yMin_(yMin), yExtent_(yExtent)
        {}

        bool ContainsZMW(uint16_t holeX, uint16_t holeY) const
        {
            return (rtConfig_->chipLayout.get()->*filter_)(holeX, holeY) &&
                   (xMin_ <= holeX && holeX < xMin_ + xExtent_) &&
                   (yMin_ <= holeY && holeY < yMin_ + yExtent_);
        }

        bool AddZmw(const PacBio::Primary::UnitCell& position, const TMetrics& bm);

        const std::string& Name() const
        { return name_; }

        template<size_t N>
        Json::Value ArrayToJson(const std::array<ContinuousDist, N>& arr, bool duplicate=false) const
        {
            Json::Value root;
            Json::Value sampleSizes(Json::arrayValue);
            Json::Value sampleMeans(Json::arrayValue);
            Json::Value sampleMedians(Json::arrayValue);
            Json::Value sampleCV(Json::arrayValue);

            size_t dupCount = (duplicate == true && N == 1) ? 2 : 1;

            for (size_t d = 0; d < dupCount; d++)
            {
                for (size_t n = 0; n < N; n++)
                {
                    sampleSizes.append(arr[n].Size());

                    if (arr[n].Size() > rtConfig_->minSampleSize)
                    {
                        sampleMeans.append(arr[n].Mean());
                        sampleMedians.append(arr[n].Median());
                        sampleCV.append(arr[n].Mean() > 0 ? arr[n].StdDev() / arr[n].Mean() : -1);
                    }
                    else
                    {
                        sampleMeans.append(-1);
                        sampleMedians.append(-1);
                        sampleCV.append(-1);
                    }
                }
            }

            root["sampleSize"] = sampleSizes;
            root["sampleMean"] = sampleMeans;
            root["sampleMed"] = sampleMedians;
            root["sampleCV"] = sampleCV;

            return root;
        }

        template<size_t N>
        std::string ArrayToCSV(const std::array<ContinuousDist, N>& arr) const
        {
            std::ostringstream csvOutput;

            Json::Value v = ArrayToJson(arr);

            for (unsigned int n = 0; n < N; n++)
            {
                csvOutput << v["sampleMean"][n].asString() << ","
                          << v["sampleMed"][n].asString() << ","
                          << v["sampleCV"][n].asString() << ","
                          << v["sampleSize"][n].asString();
                if (n + 1 < N) csvOutput << ",";
            }

            return csvOutput.str();
        }

        std::string name_;

        const ZmwMetricsRecorderConfig* rtConfig_;
        bool (PacBio::Primary::ChipLayout::*filter_)(uint16_t, uint16_t) const;
        int32_t    xMin_;
        int32_t    xExtent_;
        int32_t    yMin_;
        int32_t    yExtent_;

        ContinuousDist baseRate_;
        ContinuousDist baseWidth_;
        ContinuousDist pulseRate_;
        ContinuousDist pulseWidth_;
        AnalogsMetric  snr_;
        AnalogsMetric  pkmid_;
        ChannelsMetric baseline_;
        ChannelsMetric baselineStdDev_;
    };

    class MetricsSuperChunk
    {
    public:
        static const std::string metricsInfoStr;
    public:

        class MetricsBlock
        {
        public:
            MetricsBlock(const RTMetricsConfig::ConfigurationArray<RTMetricsRegion>& regions,
                         const ZmwMetricsRecorderConfig* rtConfig)
            : numZmws_(0)
            , sequencingZMWs_("", rtConfig, &ChipLayout::IsPorSequencing)
            , noZMWsnoAperturesZMWs_("NoZMWsNoApertures", rtConfig, &ChipLayout::IsApertureClosed)
            , noZMWsAperturesOpenZMWs_("NoZMWsAperturesOpen", rtConfig, &ChipLayout::IsApertureOpen)
            , rtConfig_(rtConfig)
            {
                for (const auto& r : regions)
                    regions_.emplace_back(r.name, rtConfig, &ChipLayout::IsPorSequencing,
                                          r.xMin, r.xExtent, r.yMin, r.yExtent);
            }

            MetricsBlock(const std::vector<ZmwRTRegionNew>& regions,
                         const ZmwMetricsRecorderConfig* rtConfig)
            : rtConfig_(rtConfig)
            {
                for (const auto& r : regions)
                    regionsNew_.emplace_back(&r, rtConfig_);
            }

            uint32_t StartFrame() const
            { return startFrame_; }

            MetricsBlock& StartFrame(uint32_t startFrame)
            {
                startFrame_ = startFrame;
                return *this;
            }

            uint16_t NumFrames() const
            { return numFrames_; }

            MetricsBlock& NumFrames(uint16_t numFrames)
            {
                numFrames_ = numFrames;
                return *this;
            }

            uint64_t StartFrameTimeStamp() const
            { return startFrameTimeStamp_; }

            uint64_t EndFrameTimeStamp() const
            { return endFrameTimeStamp_; }

            void ComputeTimeStamps(uint32_t superChunkIndex, uint64_t chunkTimeStampStart, uint64_t timeStampDelta);

            void AddZmw(const TMetrics& bm,
                        const PacBio::Primary::UnitCell& position,
                        bool filterZMW);

            uint32_t startFrame_;
            uint16_t numFrames_;
            uint64_t startFrameTimeStamp_;
            uint64_t endFrameTimeStamp_;

            uint32_t numZmws_;
            ZmwGrouping sequencingZMWs_;
            ZmwGrouping noZMWsnoAperturesZMWs_;
            ZmwGrouping noZMWsAperturesOpenZMWs_;
            std::vector<ZmwGrouping> regions_;

            std::vector<ZmwGroupingNew> regionsNew_;

            const ZmwMetricsRecorderConfig* rtConfig_;
        };  // MetricsBlock

    public:

        MetricsSuperChunk(uint32_t numMetrics, uint64_t timeStampStart, uint64_t timeStampDelta,
                          const RTMetricsConfig::ConfigurationArray<RTMetricsRegion>& regions,
                          const std::vector<ZmwRTRegionNew>& rtRegions,
                          const ZmwMetricsRecorderConfig* rtConfig,
                          std::string token)
            : numMetrics_(numMetrics)
            , timeStampStart_(timeStampStart)
            , timeStampDelta_(timeStampDelta)
            , rtConfig_(rtConfig)
            , token_(token)
        {
            if (rtConfig_->newJsonFormat)
            {
                for (size_t n = 0; n < numMetrics; n++)
                    metricsBlocks_.emplace_back(rtRegions, rtConfig_);
            }
            else
            {
                for (size_t n = 0; n < numMetrics; n++)
                    metricsBlocks_.emplace_back(regions, rtConfig_);
            }
        }

        std::vector<MetricsBlock>& MetricsBlocks()
        { return metricsBlocks_; }

        Json::Value ToJson() const;
        std::string ToCSV() const;

    private:
        uint32_t numMetrics_;
        uint64_t timeStampStart_;
        uint64_t timeStampDelta_;
        const ZmwMetricsRecorderConfig* rtConfig_;
        const std::string token_;
        std::vector<MetricsBlock> metricsBlocks_;
    }; // MetricsSuperChunk

public:
    class MetricsSuperChunkBuffer
    {
    public:

        MetricsSuperChunkBuffer(size_t initialZmwCapacity, size_t expectedZmwCapacity, size_t expectedMetricsLength)
            : numMetricsBlocks_(0)
            , superChunkIndex_(0xFFFFFFFF)
            , storage_(initialZmwCapacity * expectedMetricsLength, expectedZmwCapacity * expectedMetricsLength)
        {
            metrics_.reserve(initialZmwCapacity);
        }

        MetricsSuperChunkBuffer(size_t expectedZmwCapacity, size_t expectedMetricsLength)
            : MetricsSuperChunkBuffer(expectedZmwCapacity, expectedZmwCapacity, expectedMetricsLength) {}

        uint32_t NumMetricsBlocksPerSlice() const
        { return numMetricsBlocks_; }

        uint64_t TimeStampStart() const
        { return timeStampStart_; }

        MetricsSuperChunkBuffer& TimeStampStart(uint64_t timeStampStart)
        {
            timeStampStart_ = timeStampStart;
            return *this;
        }

        uint64_t TimeStampDelta() const
        { return timeStampDelta_; }

        MetricsSuperChunkBuffer& TimeStampDelta(uint64_t timeStampDelta)
        {
            timeStampDelta_ = timeStampDelta;
            return *this;
        }

        uint32_t SuperChunkIndex() const
        { return superChunkIndex_; }

        MetricsSuperChunkBuffer& SuperChunkIndex(uint32_t superChunkIndex)
        {
            superChunkIndex_ = superChunkIndex;
            return *this;
        }

        struct MetricsSlice
        {
            uint32_t zmwIndex;
            MemoryBufferView<TMetrics> metrics;
            MetricsSlice(uint32_t zmwIdx, 
                        const MemoryBufferView<TMetrics>& data)
                : zmwIndex(zmwIdx), metrics(data) {}

            MetricsSlice(MetricsSlice&&) = default;
        };

        const MetricsSlice& operator[](size_t idx) const
        {
            return metrics_[idx];
        }

        void AddMetricBlocks(uint32_t zmwIndex, uint32_t numMetrics, const TMetrics* metrics)
        {
            if (numMetricsBlocks_ == 0)
            {
                numMetricsBlocks_ = numMetrics;
            } else {
                assert(numMetricsBlocks_ == numMetrics);
            }
            auto data = storage_.Copy(metrics, numMetrics);
            metrics_.emplace_back(zmwIndex, data);
        }

        bool empty() const { return metrics_.empty(); }
        size_t size() const { return metrics_.size(); }

        void Reset() {
            timeStampStart_ = 0;
            timeStampDelta_ = 0;
            numMetricsBlocks_ = 0;
            superChunkIndex_ = 0;
            storage_.Reset();
            auto size = metrics_.size();
            std::vector<MetricsSlice>{}.swap(metrics_);
            metrics_.reserve(size);
        }

    private:
        uint64_t timeStampStart_;
        uint64_t timeStampDelta_;
        uint32_t numMetricsBlocks_;
        uint32_t superChunkIndex_;
        MemoryBuffer<TMetrics> storage_;
        std::vector<MetricsSlice> metrics_;
    };

public:

    // This class is used to store the parameters from
    // ConfigurationObject as plain variables since they are
    // accessed repeatedly and those objects are expensive
    // to manipulate. A raw pointer to this class
    // is stored for quick access as this class is instantatied
    // at the start of the acquisition and deleted at the end
    // of the acquisition.
    class ZmwMetricsRecorderConfig
    {
    public:
        ZmwMetricsRecorderConfig(const RTMetricsConfig& config, const Json::Value &setupJson)
        : baselineMode(config.baselineConfigMode().native())
        , signalMode(config.signalConfigMode().native())
        , minBaselineFrames(config.minBaselineFrames())
        , minSampleSize(config.minSampleSize())
        , minBaseRate(config.minBaseRate())
        , minBaseWidth(config.minBaseWidth())
        , maxBaseRate(config.maxBaseRate())
        , maxBaseWidth(config.maxBaseWidth())
        , minConfidenceScore(config.minConfidenceScore())
        , samplingFactorStride(config.samplingFactorStride())
        , newJsonFormat(config.newJsonFormat())
        , useRealtimeActivityLabels(config.useRealtimeActivityLabels())
        , maxQueueSize(config.maxQueueSize())
        {
            PacBio::Primary::Acquisition::Setup setup(setupJson);
            unitCellTypes = setup.GetUnitCellFeatureList();
            chipLayout = PacBio::Primary::ChipLayout::Factory(setup.chipLayoutName);
            frameRate = setup.expectedFrameRate;
        }

        ZmwMetricsRecorderConfig(const RTMetricsConfig& config,
                                 double frameRateIn,
                                 const std::vector<
                                         std::pair<PacBio::Primary::ChipLayout::UnitFeature, PacBio::Primary::UnitCell>> ucs,
                                 const std::string& chipLayoutName,
                                 const std::string& metricsFn)
        : baselineMode(config.baselineConfigMode().native())
        , signalMode(config.signalConfigMode().native())
        , minBaselineFrames(config.minBaselineFrames())
        , minSampleSize(config.minSampleSize())
        , minBaseRate(config.minBaseRate())
        , minBaseWidth(config.minBaseWidth())
        , maxBaseRate(config.maxBaseRate())
        , maxBaseWidth(config.maxBaseWidth())
        , minConfidenceScore(config.minConfidenceScore())
        , samplingFactorStride(config.samplingFactorStride())
        , newJsonFormat(config.newJsonFormat())
        , useRealtimeActivityLabels(config.useRealtimeActivityLabels())
        , maxQueueSize(config.maxQueueSize())
        , frameRate(frameRateIn)
        , unitCellTypes(ucs)
        , chipLayout(PacBio::Primary::ChipLayout::Factory(chipLayoutName))
        , metricsFile(metricsFn)
        {}

        RTMetricsConfig::BaselineMode::RawEnum baselineMode;
        RTMetricsConfig::SignalMode::RawEnum signalMode;
        uint32_t minBaselineFrames;
        uint32_t minSampleSize;
        float minBaseRate;
        float minBaseWidth;
        float maxBaseRate;
        float maxBaseWidth;
        float minConfidenceScore;
        uint32_t samplingFactorStride;
        bool newJsonFormat;
        bool useRealtimeActivityLabels;
        uint32_t maxQueueSize;
        double frameRate;
        std::vector<std::pair<PacBio::Primary::ChipLayout::UnitFeature, PacBio::Primary::UnitCell>> unitCellTypes;
        std::unique_ptr<PacBio::Primary::ChipLayout> chipLayout;
        std::string metricsFile = "";
    };
public: // Structors

    // Initialize with a metrics file for output for usage with the console app.
    ZmwMetricsRecorder(const RTMetricsConfig& config,
                       double frameRate,
                       const std::vector<std::pair<PacBio::Primary::ChipLayout::UnitFeature, PacBio::Primary::UnitCell>>& unitCellTypes,
                       const std::string& chipLayoutName,
                       const std::string& metricsOut,
                       const size_t maxSamples,
                       const std::string& token);

    // Initialize with publisher and setup information.
    ZmwMetricsRecorder(PacBio::IPC::MessageSocketPublisher *metricsPublisher,
                       const Json::Value& setupJson,
                       const RTMetricsConfig& config,
                       const size_t maxSamples,
                       const std::string& token);

    // Move constructor
    ZmwMetricsRecorder(ZmwMetricsRecorder &&) = delete;

    // Copy constructor
    ZmwMetricsRecorder(const ZmwMetricsRecorder &) = delete;

    // Move assignment operator
    ZmwMetricsRecorder &operator=(ZmwMetricsRecorder &&rhs) noexcept = delete;

    // Copy assignment operator
    ZmwMetricsRecorder &operator=(const ZmwMetricsRecorder &) = delete;

    ~ZmwMetricsRecorder() noexcept;

public: // Modifying methods

    bool FlushMetrics(uint32_t superChunkIndex, uint64_t timeStampStart, uint64_t timeStampDelta);

    bool AddZmwSlice(const TMetrics* hfMetrics,
                     const uint32_t numMetrics,
                     const uint32_t numBaseEvents,
                     const uint32_t zmwId);

    void WaitForTermination();

private: // Private methods

    std::string CSVHeader() const;

    void ComputeMetrics();

    bool IsSequencingZmw(const MemoryBufferView<TMetrics>& metrics);

    void StartMetricsThread();

private:

    std::unique_ptr<MetricsSuperChunkBuffer> currentMetricsSuperChunkBuffer_;
    // buffers that need processing
    PacBio::ThreadSafeQueue<std::unique_ptr<MetricsSuperChunkBuffer>> metricsSuperChunkBufferQueue_;
    // buffers that are empty and available to fill (without causing major new allocations)
    PacBio::ThreadSafeQueue<std::unique_ptr<MetricsSuperChunkBuffer>> idleMetricsSuperChunkBufferQueue_;
    std::thread sendMetricsThread_;
    bool sendMetricsThreadContinue_;
    PacBio::IPC::MessageSocketPublisher *metricsPublisher_;

    std::ofstream metricsOut_;
    bool printHeader = true;

    std::unique_ptr<ZmwMetricsRecorderConfig> rtConfig_;

    RTMetricsConfig::ConfigurationArray<RTMetricsRegion> regions_;
    std::vector<ZmwRTRegionNew> zmwRTRegions_;

    uint64_t lastChunkTimeStampStart_ = 0;
    uint64_t lastChunkTimeStampDelta_ = 0;

    // Stores ZMW ids for ZMWs that have maxed out basecall buffers indicative of basecalling issues.
    size_t maxSamples_;
    std::set<uint32_t> speedingZMWs;

    uint32_t lastSuperChunkIndex_ = 0xFFFFFFFF;
    int warningsForThisSuperChunk_ = 0;
    uint32_t numMetricsSuperChunks_ = 3;
    std::string token_;
};

using ZmwMetricsRecorderSequel = ZmwMetricsRecorder<PacBio::Primary::BasecallingMetricsT::Sequel>;
using ZmwMetricsRecorderSpider = ZmwMetricsRecorder<PacBio::Primary::BasecallingMetricsT::Spider>;

}}
