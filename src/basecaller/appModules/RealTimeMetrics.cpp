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

#include "RealTimeMetrics.h"

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;

namespace PacBio::Application
{

RealTimeMetrics::RealTimeMetrics(std::vector<Data::RealTimeMetricsRegion>& regions,
                                 std::vector<DataSourceBase::LaneSelector>& selections,
                                 std::vector<std::vector<uint32_t>>& features)
{
    for (size_t i = 0; i < regions.size(); i++)
    {
        uint32_t featuresMask = static_cast<uint32_t>(regions[i].features.front());
        for (size_t f = 1; f < regions[i].features.size(); f++)
            featuresMask |= static_cast<uint32_t>(regions[i].features[f]);
        regionInfo_.push_back({ regions[i], std::move(selections[i]),
                                SelectedLanesWithFeatures(features[i], featuresMask) });
    }
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
