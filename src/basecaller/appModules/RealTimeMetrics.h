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

#ifndef PACBIO_APPLICATION_REALTIME_METRICS_H
#define PACBIO_APPLICATION_REALTIME_METRICS_H

#include <cstdint>
#include <vector>

#include <pacbio/datasource/DataSourceRunner.h>

#include <common/graphs/GraphNodeBody.h>
#include <common/MongoConstants.h>
#include <common/LaneArray.h>
#include <common/StatAccumulator.h>
#include <dataTypes/BatchResult.h>
#include <dataTypes/configs/RealTimeMetricsConfig.h>

namespace PacBio::Application
{

class NoopRealTimeMetrics final : public Graphs::LeafBody<const Mongo::Data::BatchResult>
{
public:
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(const Mongo::Data::BatchResult& in) override
    {

    }
};

class RealTimeMetrics final: public Graphs::LeafBody<const Mongo::Data::BatchResult>
{
public:
    static std::vector<Mongo::LaneMask<>> SelectedLanesWithFeatures(const std::vector<uint32_t>& features,
                                                                    uint32_t featuresMask);
public:
    RealTimeMetrics(uint32_t framesPerHFMetricBlock, size_t numBatches,
                    const Mongo::Data::RealTimeMetricsConfig& rtConfig,
                    std::vector<DataSource::DataSourceBase::LaneSelector>&& selections,
                    const std::vector<std::vector<uint32_t>>& zmwFeatures,
                    float frameRate);
    ~RealTimeMetrics();

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(const Mongo::Data::BatchResult& in) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace PacBio::Application

#endif // PACBIO_APPLICATION_REALTIME_METRICS_H
