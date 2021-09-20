// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_APPLICATION_TRACE_SAVER_H
#define PACBIO_APPLICATION_TRACE_SAVER_H

#include <pacbio/datasource/DataSourceBase.h>
#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/SensorPacket.h>
#include <pacbio/tracefile/TraceFile.h>
#include <pacbio/logging/Logger.h>

#include <common/graphs/GraphNodeBody.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/configs/ROIConfig.h>
#include <appModules/DataFileWriter.h>
#include <pacbio/tracefile/TraceFile.h>

#include <boost/multi_array.hpp>

#include <vector>

namespace PacBio {
namespace Application {

class NoopTraceSaverBody final : public Graphs::LeafBody<const Mongo::Data::TraceVariant>
{
public:
    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(const Mongo::Data::TraceVariant&) override
    {

    }
};

class TraceSaverBody final : public Graphs::LeafBody<const Mongo::Data::TraceVariant>
{
public:
    TraceSaverBody(std::unique_ptr<PacBio::TraceFile::TraceFile>&& writer,
                   const std::vector<PacBio::DataSource::DataSourceBase::UnitCellProperties>& features,
                   PacBio::DataSource::DataSourceBase::LaneSelector&& laneSelector);

    TraceSaverBody(const TraceSaverBody&) = delete;
    TraceSaverBody(TraceSaverBody&&) = delete;
    TraceSaverBody& operator=(const TraceSaverBody&) = delete;
    TraceSaverBody& operator==(TraceSaverBody&&) = delete;

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(const Mongo::Data::TraceVariant& traceBatch) override;
private:
    std::unique_ptr<PacBio::TraceFile::TraceFile> writer_;
    PacBio::DataSource::DataSourceBase::LaneSelector laneSelector_;
};

}}

#endif //PACBIO_APPLICATION_TRACE_SAVER_H
