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
#include <pacbio/file/TraceFile.h>
#include <pacbio/logging/Logger.h>

#include <common/graphs/GraphNodeBody.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/configs/ConfigForward.h>
#include <appModules/DataFileWriter.h>

#include <boost/multi_array.hpp>

#include <vector>

namespace PacBio::Application {

// Intermediate payload containing traces, which is just a
// a multi-array for the data to be written.  The data will
// already be transposed and re-indexed, making it suitable
// for just dumping straight to disk.
//
// Note: boost::multi_array is ancient and doesn't have move
// semantics... Using `unique_ptr` as a quick workaround
template <typename T>
using RawPreppedTraces = std::unique_ptr<boost::multi_array<T, 2>>;
using PreppedTracesVariant = std::variant<RawPreppedTraces<uint8_t>, RawPreppedTraces<int16_t>>;

class TracePrepBody final : public Graphs::TransformBody<const Mongo::Data::TraceBatchVariant,
                                                         PreppedTracesVariant>
{
public:
    TracePrepBody(DataSource::DataSourceBase::LaneSelector laneSelector,
                  uint64_t maxFrames);

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.2; }

    PreppedTracesVariant Process(const Mongo::Data::TraceBatchVariant& traceBatch) override;
private:
    PacBio::DataSource::DataSourceBase::LaneSelector laneSelector_;
    uint64_t maxFrames_;
};

class TraceSaverBody final : public Graphs::LeafBody<PreppedTracesVariant>
{
public:
    TraceSaverBody(const std::string& filename,
                   uint64_t numFrames, // TODO numFrames is 64 bits but the frame indices are returned as signed 32 bit ints.
                   const uint64_t frameBlockingSize,
                   const uint64_t zmwBlockingSize,
                   File::TraceDataType dataType,
                   const std::vector<uint32_t>& holeNumbers,
                   const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                   const std::vector<uint32_t>& batchIds,
                   const File::ScanData::Data& experimentMetadata,
                   const Mongo::Data::AnalysisConfig& analysisConfig,
                   uint32_t maxQueueSize = 0);

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 0.01; }

    void Process(PreppedTracesVariant tracesVariant) override;

    ~TraceSaverBody();
private:

    void PopulateTraceData(const std::vector<uint32_t>& holeNumbers,
                           const std::vector<DataSource::DataSourceBase::UnitCellProperties>& properties,
                           const std::vector<uint32_t>& batchIds,
                           const Mongo::Data::AnalysisConfig& analysisConfig);

    void PopulateScanData(const File::ScanData::Data& experimentMetadata);

    File::TraceFile file_;
    uint64_t numFrames_;
    uint64_t numZmw_;

    std::atomic<bool> enableWriterThread = false;
    uint32_t maxQueueSize_;
    std::thread writer_;
    ThreadSafeQueue<PreppedTracesVariant> queue;
};

}

#endif //PACBIO_APPLICATION_TRACE_SAVER_H
