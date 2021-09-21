// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_APPLICATION_TRACE_DATA_SOURCE_H
#define PACBIO_APPLICATION_TRACE_DATA_SOURCE_H

#include <cstddef>
#include <cstdint>
#include <string>

#include <pacbio/datasource/DataSourceBase.h>
#include <pacbio/datasource/SensorPacketsChunk.h>
#include <pacbio/tracefile/TraceFile.h>

#include <common/BatchDataSource.h>

namespace PacBio {
namespace Application {

class TraceFileDataSource : public Mongo::BatchDataSource
{
public:
    // Reanalysis ctor.  We'll pull data dimensions from the trace file
    TraceFileDataSource(DataSourceBase::Configuration config,
                        std::string file)
        : TraceFileDataSource(std::move(config), file, 0, 0, false, 0, 0)
    {
        // Need to update the layouts_ member to know how to extract the
        // required info from the tracefile
        PBLOG_WARN << "TraceFile Re-Analysis not yet fully supported. "
                   << "Original ZMW poolIDs are not yet preserved";
    }

    // Performance testing ctor.  Data dimensions are specified and data
    // will be replicated as necessary.  Caching and preloading options are
    // also provided, to help remove file IO as a bottleneck.
    TraceFileDataSource(DataSourceBase::Configuration cfg,
                        std::string file,
                        uint32_t frames,
                        uint32_t numZmw,
                        bool cache,
                        size_t preloadChunks = 0,
                        size_t maxQueueSize = 0);

public:

    void ContinueProcessing() override;

public:

    std::vector<uint32_t> UnitCellIds() const override;
    std::vector<UnitCellProperties> GetUnitCellProperties() const override;
    std::map<uint32_t, DataSource::PacketLayout> PacketLayouts() const override
    {
        return layouts_;
    }

    size_t BlockWidth() const { return GetConfig().requestedLayout.BlockWidth(); }
    size_t BlockLen() const { return GetConfig().requestedLayout.NumFrames(); }

    size_t NumChunks() const { return numChunks_; }
    size_t NumZmwLanes() const { return numZmwLanes_; }
    size_t NumTraceChunks() const { return numTraceChunks_ ; }
    size_t NumTraceLanes() const { return numTraceLanes_; }
    size_t NumTraceZmws() const { return numTraceZmws_; }
    size_t NumTraceFrames() const { return numTraceFrames_; }

    size_t NumFrames() const override { return numChunks_ * BlockLen(); }
    size_t NumZmw() const override { return numZmwLanes_ * BlockWidth(); }
    double FrameRate() const override { return frameRate_; }

    int16_t Pedestal() const override { return this->traceFile_.Traces().Pedestal(); }

    DataSource::HardwareInformation GetHardwareInformation() override
    {
        PacBio::DataSource::HardwareInformation info;
        PacBio::DataSource::Version one;
        one.major = 1;
        one.minor = 0;
        one.build = 0;
        one.revision = 0;
        one.branch = "develop"; // FIXME
        info.driverVersion = one;
        info.fwVersion = one;
        info.hwVersion = one;
        info.swVersion = one;
        info.SetShortName( "TraceFileDataSource");
        info.SetSummary("Tracefile DataSource using " + filename_ + " as input");
        info.SetSummary( info.ShortName() + " " + info.hwVersion.ToString());
        return info;
    }


private:

    // throw a bunch of data into the queues during construction rather than after
    // a thread is spawned.  Can greatly increase both startup time and memory footprint,
    // but does allow data processing guaranteed to not have any IO bottlenecking
    void PreloadInputQueue(size_t chunks);
    void PopulateBlock(size_t traceLane, size_t traceChunk, uint8_t* data);
    template <typename T>
    void ReadBlockFromTraceFile(size_t traceLane, size_t traceChunk, T* data);

    size_t numZmwLanes_;
    size_t numChunks_;
    size_t numTraceZmws_;
    size_t numTraceFrames_;
    size_t numTraceLanes_;
    size_t numTraceChunks_;
    size_t chunkIndex_;
    size_t batchIndex_;
    size_t currZmw_;
    size_t maxQueueSize_;

    float frameRate_;
    uint32_t bytesPerValue_;

    std::string filename_;

    TraceFile::TraceFile traceFile_;
    std::vector<uint8_t> traceDataCache_;
    std::vector<size_t> laneCurrentChunk_;
    bool cache_;
    DataSource::SensorPacketsChunk currChunk_;

    std::map<uint32_t, DataSource::PacketLayout> layouts_;
};

}}

#endif //PACBIO_APPLICATION_TRACE_DATA_SOURCE_H
