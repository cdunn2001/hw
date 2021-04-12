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
    , currZmw_{0}
    , maxQueueSize_(maxQueueSize)
    , filename_(file)
    , traceFile_(filename_)
    , cache_(cache)
    , currChunk_(0, GetConfig().requestedLayout.NumFrames())
{
    const auto& config = GetConfig();
    if (config.darkFrame != nullptr)
        throw PBException("Dark frame subtraction not currently supported for trace files");
    if (config.crosstalkFilter != nullptr)
        throw PBException("Cross talk correction not currently supported for trace files");
    if (config.decimationMask != nullptr)
        throw PBException("Decimation mask not currently supported for trace files");

    // SPARSE will be appropriate to allow once we fully support TraceFile Re-Analysis
    if (config.requestedLayout.Type() != PacketLayout::BLOCK_LAYOUT_DENSE)
        throw PBException("Trace file source currently only supports dense block layout");
    if (config.requestedLayout.Encoding() != PacketLayout::INT16)
        throw PBException("Trace file source currently only supports 16 bit encoding");

    if (config.requestedLayout.BlockWidth() != laneSize)
        throw PBException("Unexpected lane width requested");

    if (maxQueueSize_ == 0) maxQueueSize_ = preloadChunks + 1;

    // TODO should be able to handle all types eventually
    assert(config.requestedLayout.Type() == PacketLayout::BLOCK_LAYOUT_DENSE);
    assert(config.requestedLayout.Encoding() == PacketLayout::INT16);
    assert(config.requestedLayout.BlockWidth() == laneSize);

    // Could potentially round up to the nearest lane size, but no matter
    // what we require 64 zmw lanes, both in the trace file and in our
    // analysis pipeline
    if (numZmw % config.requestedLayout.BlockWidth() != 0)
    {
        throw PBException("Requested numZmw must be an even multiple of laneSize (64)");
    }

    numZmwLanes_ = numZmw / (config.requestedLayout.BlockWidth());
    numChunks_ = (frames + BlockLen() - 1) / BlockLen();

    const auto numFullPools = numZmwLanes_ / config.requestedLayout.NumBlocks();
    for (size_t i = 0; i < numFullPools; ++i)
    {
        layouts_[i] = config.requestedLayout;
    }
    size_t stubBlocks = numZmwLanes_ % config.requestedLayout.NumBlocks();
    if (stubBlocks != 0)
    {
        PacketLayout stub(config.requestedLayout.Type(),
                          config.requestedLayout.Encoding(),
                          {stubBlocks,
                           config.requestedLayout.NumFrames(),
                           config.requestedLayout.BlockWidth()});
        layouts_[numFullPools] = stub;
    }

    frameRate_ = traceFile_.Scan().AcqParams().frameRate;
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

    uint32_t traceStartZmwLane = currZmw_ / BlockWidth();
    uint32_t wrappedChunkIndex = chunkIndex_ % NumTraceChunks();

    const auto& currLayout = layouts_[batchIndex_];
    const auto startFrame = chunkIndex_ * BlockLen();
    SensorPacket batchData(currLayout, batchIndex_, currZmw_, startFrame, *GetConfig().allocator);

    for (size_t lane = 0; lane < currLayout.NumBlocks(); lane++)
    {
        auto block = batchData.BlockData(lane);
        assert(block.Count() * BlockWidth()*BlockLen()*sizeof(int16_t));

        uint32_t wrappedLane = (traceStartZmwLane + lane) % NumTraceLanes();
        PopulateBlock(wrappedLane, wrappedChunkIndex, reinterpret_cast<int16_t*>(block.Data()));
    }

    currChunk_.AddPacket(std::move(batchData));

    batchIndex_++;
    currZmw_ += currLayout.NumZmw();
    if (currZmw_ == NumZmw())
    {
        auto chunk = SensorPacketsChunk(currChunk_.StopFrame(), currChunk_.StopFrame() + BlockLen(), layouts_.size());
        std::swap(chunk, currChunk_);
        this->PushChunk(std::move(chunk));
        batchIndex_ = 0;
        currZmw_ = 0;
        chunkIndex_++;
    }

    if (chunkIndex_ == NumChunks())
        this->SetDone();
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
    for(uint32_t i=0; i < features.size(); i++)
    {
        // i is ZMW position in chunk.
        // The chunk is filled with lanes from the tracefile, modulo the size of the trace file. It is 
        // possible that the trace file ZMW count is not modulo the BlockWidth. In other words, the final
        // ZMWs could be ragged in the last trace file lane. So to calculate the ZMW position in the trace file,
        // we have to down convert to chunk lanes, then modulo that with the number of tracelanes, then scale back up.
        // Thankfully the blocks are the same size in the chunk as in the trace file.
        const auto chunkLane = i / BlockWidth();
        const auto offset = i % BlockWidth();
        const auto traceLane = chunkLane % NumTraceLanes();
        const auto traceZmw = traceLane * BlockWidth() + offset; // position in trace file

        if (traceZmw >= holexy.shape()[0])
        {
            throw PBException("internally calculated traceZmw position is larger than trace file dimension");
        }
        features[i].flags = 0; // fixme
        features[i].x = holexy[traceZmw][0];
        features[i].y = holexy[traceZmw][1];
        // features[i].holeNumber = holeNumber[traceZmw]; TODO
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

