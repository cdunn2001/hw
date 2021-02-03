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

#include <appModules/TraceFileDataSource.h>

#include <numeric>

#include <pacbio/logging/Logger.h>

#include <common/MongoConstants.h>

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;

namespace PacBio {
namespace Application {

TraceFileDataSource::TraceFileDataSource(
        DataSourceBase::Configuration cfg,
        std::string file,
        uint32_t frames,
        uint32_t numZmw,
        bool cache,
        size_t preloadChunks,
        size_t maxQueueSize)
    : DataSourceBase(std::move(cfg))
    , chunkIndex_{0}
    , batchIndex_{0}
    , maxQueueSize_(maxQueueSize)
    , filename_(file)
    , traceFile_(filename_)
    , cache_(cache)
    , currChunk_(0, GetConfig().layout.NumFrames())
{
    const auto& config = GetConfig();
    if (config.darkFrame != nullptr)
        throw PBException("Dark frame subtraction not currently supported for trace files");
    if (config.crosstalkFilter != nullptr)
        throw PBException("Cross talk correction not currently supported for trace files");
    if (config.decimationMask != nullptr)
        throw PBException("Decimation mask not currently supported for trace files");

    if (config.layout.Type() != PacketLayout::BLOCK_LAYOUT_DENSE)
        throw PBException("Trace file source currently only supports dense block layout");
    if (config.layout.Encoding() != PacketLayout::INT16)
        throw PBException("Trace file source currently only supports 16 bit encoding");

    if (config.layout.BlockWidth() != laneSize)
        throw PBException("Unexpected lane width requested");

    if (maxQueueSize_ == 0) maxQueueSize_ = preloadChunks + 1;

    // TODO should be able to handle both eventually
    assert(config.layout.Type() == PacketLayout::BLOCK_LAYOUT_DENSE);
    assert(config.layout.Encoding() == PacketLayout::INT16);
    assert(config.layout.BlockWidth() == laneSize);

    // TODO this restriction should be lifted
    assert(numZmw % config.layout.BlockWidth() == 0);
    assert(numZmw / config.layout.BlockWidth() % config.layout.NumBlocks() == 0);

    numZmwLanes_ = numZmw / (config.layout.BlockWidth());
    numChunks_ = (frames + BlockLen() - 1) / BlockLen();

    numTraceZmws_ = traceFile_.Traces().NumZmws();
    numTraceFrames_ = traceFile_.Traces().NumFrames();
    numTraceLanes_ = (numTraceZmws_ + BlockWidth() - 1) / BlockWidth();
    numTraceChunks_ = (numTraceFrames_ + BlockLen() - 1) / BlockLen();

    if (numZmwLanes_ == 0) numZmwLanes_ = numTraceLanes_;
    if (numChunks_ == 0) numChunks_ = numTraceChunks_;

    // Adjust number of lanes and chunks we read from the trace file
    // if requested lanes and chunks is less to only cache what we need.
    numTraceLanes_ = std::min(numZmwLanes_, numTraceLanes_);
    numTraceChunks_ = std::min(numChunks_, numTraceChunks_);

    if (cache_)
    {
        // Cache requested portion of trace file into memory.
        traceDataCache_.resize(NumTraceLanes()*BlockWidth()*NumTraceChunks()*BlockLen());
        for (size_t traceLane = 0; traceLane < NumTraceLanes(); traceLane++)
        {
            for (size_t traceChunk = 0; traceChunk < NumTraceChunks(); traceChunk++)
            {
                ReadBlockFromTraceFile(traceLane, traceChunk,
                                       traceDataCache_.data() +
                                       (traceLane*BlockWidth()*BlockLen()*NumTraceChunks()) +
                                       (traceChunk*BlockWidth()*BlockLen()));
            }
        }
    }
    else
    {
        // Maintain cache of blocks for current active chunk to support replicating in ZMW space.
        traceDataCache_.resize(NumTraceLanes()*BlockWidth()*BlockLen());
        laneCurrentChunk_.resize(NumTraceLanes(), std::numeric_limits<size_t>::max());
    }

    if (preloadChunks != 0) PreloadInputQueue(preloadChunks);
}

void TraceFileDataSource::ContinueProcessing()
{
    if (ChunksReady() >= maxQueueSize_) return;

    uint32_t traceStartZmwLane = (batchIndex_ * BatchLanes()) % NumTraceLanes();
    uint32_t wrappedChunkIndex = chunkIndex_ % NumTraceChunks();

    const auto startZmw = batchIndex_ * BatchLanes() * BlockWidth();
    const auto startFrame = chunkIndex_ * BlockLen();
    SensorPacket batchData(GetConfig().layout, startZmw, startFrame, *GetConfig().allocator);

    for (size_t lane = 0; lane < BatchLanes(); lane++)
    {
        auto block = batchData.BlockData(lane);
        assert(block.Count() * BlockWidth()*BlockLen()*sizeof(int16_t));

        uint32_t wrappedLane = (traceStartZmwLane + lane) % NumTraceLanes();
        PopulateBlock(wrappedLane, wrappedChunkIndex, reinterpret_cast<int16_t*>(block.Data()));
    }

    currChunk_.AddPacket(std::move(batchData));

    batchIndex_++;
    if (batchIndex_ == NumBatches())
    {
        auto chunk = SensorPacketsChunk(currChunk_.StopFrame(), currChunk_.StopFrame() + BlockLen(), NumBatches());
        std::swap(chunk, currChunk_);
        this->PushChunk(std::move(chunk));
        batchIndex_ = 0;
        chunkIndex_++;
    }

    if (chunkIndex_ == NumChunks())
        this->SetDone();
}

std::vector<uint32_t> TraceFileDataSource::PoolIds() const
{
    // TODO needs to change to support sparse data
    std::vector<uint32_t> poolIds(NumBatches());
    for (size_t i = 0; i < NumBatches(); ++i) poolIds[i] = i;
    return poolIds;
}

std::vector<uint32_t> TraceFileDataSource::UnitCellIds() const
{
    std::vector<uint32_t> unitCellNumbers(numZmwLanes_ * BlockWidth());
    std::iota(unitCellNumbers.begin(), unitCellNumbers.end(), 0);
    return unitCellNumbers;
}

std::vector<DataSourceBase::UnitCellProperties> TraceFileDataSource::GetUnitCellProperties() const
{
    std::vector<DataSourceBase::UnitCellProperties> features(numZmwLanes_ * BlockWidth());
    boost::multi_array<int16_t,2> holexy = traceFile_.Traces().HoleXY(); 
    for(uint32_t i=0; i < holexy.shape()[0]; i++)
    {
        features[i].flags = 0; // fixme
        features[i].x = holexy[i][0];
        features[i].y = holexy[i][1];
    }
    return features;
}

void TraceFileDataSource::PreloadInputQueue(size_t chunks)
{
    size_t numPreload = std::min(chunks, NumChunks());

    if (numPreload > 0)
    {
        PBLOG_INFO << "Preloading input data queue with " + std::to_string(numPreload) + " chunks";
        while (chunkIndex_ < numPreload)
        {
            if (batchIndex_ == 0)
                PBLOG_INFO << "Preloading chunk " << chunkIndex_;
            ContinueProcessing();
        }
        PBLOG_INFO << "Done preloading input queue.";
    }
}

void TraceFileDataSource::PopulateBlock(size_t traceLane, size_t traceChunk, int16_t* data)
{
    if (cache_)
    {
        std::memcpy(data,
                    traceDataCache_.data() +
                    (traceLane*BlockWidth()*BlockLen()*NumTraceChunks()) +
                    (traceChunk*BlockWidth()*BlockLen()),
                    BlockLen()*BlockWidth()*sizeof(int16_t));
    }
    else
    {
        if (laneCurrentChunk_[traceLane] != traceChunk)
        {
            ReadBlockFromTraceFile(traceLane, traceChunk,
                                   traceDataCache_.data()+(traceLane*BlockWidth()*BlockLen()));
            laneCurrentChunk_[traceLane] = traceChunk;
        }
        std::memcpy(data, traceDataCache_.data()+(traceLane*BlockWidth()*BlockLen()),
                    BlockWidth()*BlockLen()*sizeof(int16_t));
    }
}

void TraceFileDataSource::ReadBlockFromTraceFile(size_t traceLane, size_t traceChunk, int16_t* data)
{
    size_t nZmwsToRead = std::min(BlockWidth(), NumTraceZmws() - (traceLane*BlockWidth()));
    size_t nFramesToRead = std::min(BlockLen(), NumTraceFrames() - (traceChunk*BlockLen()));
    using range = boost::multi_array_types::extent_range;
    const range zmwRange(traceLane*BlockWidth(), (traceLane*BlockWidth()) + nZmwsToRead);
    const range frameRange(traceChunk*BlockLen(), (traceChunk*BlockLen()) + nFramesToRead);
    boost::multi_array<int16_t,2> d{boost::extents[zmwRange][frameRange]};
    boost::multi_array_ref<int16_t,2> out{data, boost::extents[nFramesToRead][nZmwsToRead]};
    traceFile_.Traces().ReadTraceBlock(d);
    d.reindex(0);
    for (size_t zmw = 0; zmw < nZmwsToRead; zmw++)
    {
        for (size_t frame = 0; frame < nFramesToRead; frame++)
        {
            out[frame][zmw] = d[zmw][frame];
        }
    }
    if (nZmwsToRead*nFramesToRead < BlockWidth()*BlockLen())
    {
        for (size_t frame = nFramesToRead; frame < BlockLen(); frame++)
        {
            for (size_t zmw = nZmwsToRead; zmw < BlockWidth(); zmw++)
            {
                out[frame][zmw] = 0;
            }
        }
    }
}

}}

