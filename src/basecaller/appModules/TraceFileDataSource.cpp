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

namespace {

size_t DivideWithCeil(size_t numerator, size_t denominator)
{
    return (numerator + denominator - 1) / denominator;
}

// 0 lanes requested is a special value meaning use exactly as many lanes as
// the tracefile contains (rounding up to an even lane width)
size_t ComputeTraceLanes(size_t numTraceZmw, size_t laneWidth, size_t lanesRequested)
{
    auto defaultVal = DivideWithCeil(numTraceZmw, laneWidth);
    if (lanesRequested != 0 && lanesRequested < defaultVal)
        return lanesRequested;
    else
        return defaultVal;
}

// 0 frames requested is a special value meaning use exactly as many frames as
// the tracefile contains (rounding up to an even chunk size)
size_t ComputeTraceChunks(size_t numTraceFrames, size_t blockLen, size_t framesRequested)
{
    auto defaultVal = DivideWithCeil(numTraceFrames, blockLen);
    auto chunksRequested = DivideWithCeil(framesRequested, blockLen);
    if (chunksRequested != 0 && chunksRequested < defaultVal)
        return chunksRequested;
    else
        return defaultVal;
}

PacketLayout::EncodingFormat ComputeEncoding(PacBio::Mongo::Data::TraceInputType requestedType,
                                             TraceFile::TraceDataType storageType)
{
    PacketLayout::EncodingFormat ret;
    switch (requestedType)
    {
    case PacBio::Mongo::Data::TraceInputType::Natural:
    {
        if (storageType == TraceFile::TraceDataType::INT16)
            ret = PacketLayout::INT16;
        else if (storageType == TraceFile::TraceDataType::UINT8)
            ret = PacketLayout::UINT8;
        else
            throw PBException("Unexpected request for trace data type");
        break;
    }
    case PacBio::Mongo::Data::TraceInputType::INT16:
    {
        ret = PacketLayout::INT16;
        break;
    }
    case PacBio::Mongo::Data::TraceInputType::UINT8:
    {
        ret = PacketLayout::UINT8;
        break;
    }
    default:
        throw PBException("Unexpected request for trace data type");
    }

    if (storageType == TraceFile::TraceDataType::INT16
        && ret == PacketLayout::UINT8)
    {
        PBLOG_WARN << "Trace data is 16 bit but we are configured to produce 8 bit data.  "
                   << "Values will be saturated to [0,255]";
    }

    return ret;
}

}

TraceFileDataSource::TraceFileDataSource(
        DataSourceBase::Configuration cfg,
        std::string file,
        uint32_t frames,
        uint32_t numZmwLanes,
        bool cache,
        size_t preloadChunks,
        size_t maxQueueSize,
        Mongo::Data::TraceInputType type)
    : BatchDataSource(std::move(cfg))
    , filename_(file)
    , traceFile_(filename_)
    , numTraceZmws_(traceFile_.Traces().NumZmws())
    , numTraceFrames_(traceFile_.Traces().NumFrames())
    , numTraceLanes_(ComputeTraceLanes(numTraceZmws_, BlockWidth(), numZmwLanes))
    , numTraceChunks_(ComputeTraceChunks(numTraceFrames_, BlockLen(), frames))
    , frameRate_(traceFile_.Scan().AcqParams().frameRate)
    , numZmwLanes_(numZmwLanes == 0 ? numTraceLanes_ : numZmwLanes)
    , numChunks_(frames == 0 ? numTraceChunks_ : DivideWithCeil(frames, BlockLen()))
    , maxQueueSize_(maxQueueSize == 0 ? preloadChunks + 1 : maxQueueSize)
    , cache_(cache)
    , currChunk_(0, BlockLen())
{
    const auto& config = GetConfig();
    if (config.darkFrame != nullptr)
        throw PBException("Dark frame subtraction not currently supported for trace files");
    if (config.crosstalkFilter != nullptr)
        throw PBException("Cross talk correction not currently supported for trace files");
    if (config.decimationMask != nullptr)
        throw PBException("Decimation mask not currently supported for trace files");

    auto storageType = traceFile_.Traces().StorageType();

    // BENTODO tweak this for re-analysis
    const auto layoutType = PacketLayout::BLOCK_LAYOUT_DENSE;
    const auto encoding = ComputeEncoding(type, storageType);

    bytesPerValue_ = [&]()
    {
        switch(encoding)
        {
        case PacBio::DataSource::PacketLayout::INT16:
            return 2;
        case PacBio::DataSource::PacketLayout::UINT8:
            return 1;
        default:
            throw PBException("Unsupported encoding");
        }
    }();

    if (config.requestedLayout.BlockWidth() != laneSize)
        throw PBException("Unexpected lane width requested");

    const auto numFullPools = numZmwLanes_ / config.requestedLayout.NumBlocks();
    PacketLayout regLayout(layoutType,
                           encoding,
                           {config.requestedLayout.NumBlocks(),
                            config.requestedLayout.NumFrames(),
                            config.requestedLayout.BlockWidth()});
    for (size_t i = 0; i < numFullPools; ++i)
    {
        layouts_[i] = regLayout;
    }
    size_t stubBlocks = numZmwLanes_ % config.requestedLayout.NumBlocks();
    if (stubBlocks != 0)
    {
        PacketLayout stub(layoutType,
                          encoding,
                          {stubBlocks,
                           config.requestedLayout.NumFrames(),
                           config.requestedLayout.BlockWidth()});
        layouts_[numFullPools] = stub;
    }

    if (cache_)
    {
        // Cache requested portion of trace file into memory.
        traceDataCache_.resize(boost::extents[NumTraceChunks()][NumTraceLanes()][BlockWidth() * BlockLen() * bytesPerValue_]);
        for (size_t traceLane = 0; traceLane < NumTraceLanes(); traceLane++)
        {
            for (size_t traceChunk = 0; traceChunk < NumTraceChunks(); traceChunk++)
            {
                auto* ptr = traceDataCache_[traceChunk][traceLane].origin();
                switch(encoding)
                {
                case PacBio::DataSource::PacketLayout::INT16:
                    ReadBlockFromTraceFile(traceLane, traceChunk, reinterpret_cast<int16_t*>(ptr));
                    break;
                case PacBio::DataSource::PacketLayout::UINT8:
                    ReadBlockFromTraceFile(traceLane, traceChunk, ptr);
                    break;
                default:
                    throw PBException("Unsupported Encoding");
                }
            }
        }
    }
    else
    {
        // Maintain cache of blocks for current active chunk to support replicating in ZMW space.
        traceDataCache_.resize(boost::extents[1][NumTraceLanes()][BlockWidth()*BlockLen()*bytesPerValue_]);
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
        assert(block.Count() == BlockWidth()*BlockLen()*bytesPerValue_);

        uint32_t wrappedLane = (traceStartZmwLane + lane) % NumTraceLanes();
        PopulateBlock(wrappedLane, wrappedChunkIndex, block.Data());
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

void TraceFileDataSource::PopulateBlock(size_t traceLane, size_t traceChunk, uint8_t* data)
{
    if (cache_)
    {
        std::memcpy(data,
                    traceDataCache_[traceChunk][traceLane].origin(),
                    BlockLen()*BlockWidth()*bytesPerValue_);
    }
    else
    {
        if (laneCurrentChunk_[traceLane] != traceChunk)
        {
            auto* ptr = traceDataCache_[0][traceLane].origin();
            switch(layouts_.begin()->second.Encoding())
            {
            case PacBio::DataSource::PacketLayout::INT16:
                ReadBlockFromTraceFile(traceLane, traceChunk, reinterpret_cast<int16_t*>(ptr));
                break;
            case PacBio::DataSource::PacketLayout::UINT8:
                ReadBlockFromTraceFile(traceLane, traceChunk, ptr);
                break;
            default:
                throw PBException("Unsupported Encoding");
            }
            laneCurrentChunk_[traceLane] = traceChunk;
        }
        std::memcpy(data, traceDataCache_.data()+(traceLane*BlockWidth()*BlockLen())*bytesPerValue_,
                    BlockWidth()*BlockLen()*bytesPerValue_);
    }
}

template <typename T>
void TraceFileDataSource::ReadBlockFromTraceFile(size_t traceLane, size_t traceChunk, T* data)
{
    size_t nZmwsToRead = std::min(BlockWidth(), NumTraceZmws() - (traceLane*BlockWidth()));
    size_t nFramesToRead = std::min(BlockLen(), NumTraceFrames() - (traceChunk*BlockLen()));
    using range = boost::multi_array_types::extent_range;
    const range zmwRange(traceLane*BlockWidth(), (traceLane*BlockWidth()) + nZmwsToRead);
    const range frameRange(traceChunk*BlockLen(), (traceChunk*BlockLen()) + nFramesToRead);
    boost::multi_array<T,2> d{boost::extents[zmwRange][frameRange]};
    boost::multi_array_ref<T,2> out{data, boost::extents[nFramesToRead][nZmwsToRead]};
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

TraceFileDataSource::LaneSelector TraceFileDataSource::SelectedLanesWithinROI(const std::vector<std::vector<int>>& vec) const
{
    if (vec.empty())
    {
        std::vector<LaneIndex> dummy(0);
        return LaneSelector(dummy);
    }

    if (reanalysis_)
        throw PBException("Trace ReAnalysis does not currently support selecting an ROI for trace saving");

    std::set<LaneIndex> selected;
    for (const auto& range : vec)
    {
        if (range.size() == 0 || range.size() > 2)
            throw PBException("Unexpected format for TraceFileDataSource ROI.  "
                              "The inner most vector should be a single element "
                              "representing a ZMW, or two values representing a start ZMW and count");

        // always going to enter at least one lane, corresponding to the first element.
        // This first ZMW may be in the middle of a lane, but we'll add the whole lane
        // anyway
        selected.insert(range[0]/laneSize);
        if (range.size() == 2)
        {
            // We've already addd the first lane, now we use some intentional
            // integer arithmetic to get the rest of the lanes, even if the ROI
            // specified doesn't line up with lane boundaries.
            int laneStart = (range[0] + laneSize) / laneSize;
            int laneEnd = (range[1] - 1) / laneSize;
            for (int i = laneStart; i <= laneEnd; ++i)
            {
                selected.insert(i);
            }
        }
    }
    std::vector<LaneIndex> retValues(selected.begin(), selected.end());
    return LaneSelector(retValues);
}


}}
