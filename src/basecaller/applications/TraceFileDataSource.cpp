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

#include <applications/TraceFileDataSource.h>

#include <numeric>

#include <pacbio/logging/Logger.h>

#include <common/MongoConstants.h>

using namespace PacBio::DataSource;
using namespace PacBio::Mongo;
using namespace PacBio::Cuda::Data;

namespace PacBio {
namespace Application {

// size_t numZmwLanes_;
// size_t numChunks_;
// size_t numTraceChunks_;
// size_t chunkIndex_;
// size_t batchIndex_;
// size_t maxQueueSize_;

// std::unique_ptr<Cuda::Data::TraceFileReader> traceFileReader_;
// DataSource::SensorPacketsChunk currChunk_;
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

    if (maxQueueSize_ == 0) maxQueueSize_ = preloadChunks + 2;

    // TODO should be able to handle both eventually
    assert(config.layout.Type() == PacketLayout::BLOCK_LAYOUT_DENSE);
    assert(config.layout.Encoding() == PacketLayout::INT16);
    assert(config.layout.BlockWidth() == laneSize);

    // TODO this restriction should be lifted
    assert(numZmw % config.layout.BlockWidth() == 0);
    assert(numZmw / config.layout.BlockWidth() % config.layout.NumBlocks() == 0);

    numZmwLanes_ = numZmw / (config.layout.BlockWidth());
    numChunks_ = (frames + BlockLen() - 1) / BlockLen();

    traceFileReader_ = std::make_unique<TraceFileReader>(file, BlockWidth(), BlockLen(), cache);
    if (numZmwLanes_ == 0) numZmwLanes_ = traceFileReader_->NumZmwLanes();
    if (numChunks_ == 0) numChunks_ = traceFileReader_->NumChunks();
    numTraceChunks_ = traceFileReader_->NumChunks();

    if (preloadChunks != 0) PreloadInputQueue(preloadChunks);
}

void TraceFileDataSource::ContinueProcessing()
{
    uint32_t traceStartZmwLane = (batchIndex_ * BatchLanes()) % traceFileReader_->NumZmwLanes();
    uint32_t wrappedChunkIndex = chunkIndex_ % traceFileReader_->NumChunks();

    const auto startZmw = batchIndex_ * BatchLanes() * BlockWidth();
    const auto startFrame = chunkIndex_ * BlockLen();
    SensorPacket batchData(GetConfig().layout, startZmw, startFrame, *GetConfig().allocator);

    for (size_t lane = 0; lane < BatchLanes(); lane++)
    {
        auto block = batchData.BlockData(lane);
        assert(block.Count() * BlockWidth()*BlockLen()*sizeof(int16_t));

        uint32_t wrappedLane = (traceStartZmwLane + lane) % traceFileReader_->NumZmwLanes();
        traceFileReader_->PopulateBlock(wrappedLane, wrappedChunkIndex, reinterpret_cast<int16_t*>(block.Data()));
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

std::vector<uint32_t> TraceFileDataSource::UnitCellFeatures() const
{
    return std::vector<uint32_t>(numZmwLanes_ * BlockWidth(), 0);
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

}}

