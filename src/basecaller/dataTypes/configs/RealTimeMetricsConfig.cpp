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

#include <dataTypes/configs/RealTimeMetricsConfig.h>

namespace PacBio::Mongo::Data
{

static void AddSignalMetrics(std::vector<MetricNames>* metrics)
{
    metrics->push_back(MetricNames::Baseline);
    metrics->push_back(MetricNames::BaselineStd);
    metrics->push_back(MetricNames::Pkmid);
    metrics->push_back(MetricNames::Snr);
}

static void AddKineticMetrics(std::vector<MetricNames>* metrics)
{
    metrics->push_back(MetricNames::PulseRate);
    metrics->push_back(MetricNames::PulseWidth);
    metrics->push_back(MetricNames::BaseRate);
    metrics->push_back(MetricNames::BaseWidth);
}

static void AddAllMetrics(std::vector<MetricNames>* metrics)
{
    AddKineticMetrics(metrics);
    AddSignalMetrics(metrics);
}


std::vector<RealTimeMetricsRegion> DefaultKestrelRegions()
{
    std::vector<RealTimeMetricsRegion> regions;

    {
        auto& r = regions.emplace_back();
        r.name = "FullChip";

        r.medianIntraLaneStride = 8;
        r.featuresForFilter.push_back(DataSource::ZmwFeatures::PorSequencing);
        AddAllMetrics(&r.metrics);
        r.roi.push_back({0,0,4096,6144});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "TopStrip";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::PorSequencing);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({200, 64, 64, 6016});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "MidStrip";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::PorSequencing);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({2016, 64, 64, 6016});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "BotStrip";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::PorSequencing);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({3832, 64, 64, 6016});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "FullChip_LP2P0X";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::LaserPower2p0x);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({0,0,4096,6144});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "FullChip_LP1P5X";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::LaserPower1p5x);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({0,0,4096,6144});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "FullChip_LP0P5X";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::LaserPower0p5x);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({0,0,4096,6144});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "FullChip_LP0P0X";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::LaserPower0p0x);
        AddSignalMetrics(&r.metrics);
        r.roi.push_back({0,0,4096,6144});
    }

    {
        auto& r = regions.emplace_back();
        r.name = "FullChip_LS";

        r.featuresForFilter.push_back(DataSource::ZmwFeatures::LaserScatter);
        r.metrics.push_back(MetricNames::Baseline);
        r.metrics.push_back(MetricNames::BaselineStd);
        r.useSingleActivityLabels = false;
        r.roi.push_back({0,0,4096,6144});
    }

    return regions;
}

}
