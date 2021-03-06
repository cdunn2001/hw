#ifndef BATCH_GENERATOR_H
#define BATCH_GENERATOR_H

#include <common/DataGenerators/GeneratorBase.h>
#include <common/DataGenerators/TraceFileReader.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/TraceBatch.h>
#include <numeric>

namespace PacBio {
namespace Cuda {
namespace Data {

class BatchGenerator : public GeneratorBase<int16_t>
{
public:
    BatchGenerator(uint32_t blockLen, uint32_t laneWidth, uint32_t kernelLanes,
                   uint32_t frames, uint32_t numZmwLanes)
        : GeneratorBase(blockLen,
                        laneWidth,
                        (frames + blockLen - 1)/blockLen,
                        numZmwLanes)
        , laneWidth_(laneWidth)
        , kernelLanes_(kernelLanes)
        , numZmwLanes_(numZmwLanes)
        , numChunks_(NumBlocks())
        , numTraceChunks_(0)
        , chunkIndex_(0)
    {}

    std::vector<uint32_t> UnitCellIds()
    {
        std::vector<uint32_t> unitCellNumbers(numZmwLanes_ * laneWidth_);
        std::iota(unitCellNumbers.begin(), unitCellNumbers.end(), 0);
        return unitCellNumbers;
    }

    std::vector<uint32_t> UnitCellFeatures()
    {
        return std::vector<uint32_t>(numZmwLanes_ * laneWidth_, 0);
    }

    void SetTraceFileSource(const std::string& traceFileName, bool cache)
    {
        traceFileReader_.reset(new TraceFileReader(traceFileName, laneWidth_, BlockLen(), cache));
        if (numZmwLanes_ == 0) numZmwLanes_ = traceFileReader_->NumZmwLanes();
        if (numChunks_ == 0) numChunks_ = traceFileReader_->NumChunks();
        numTraceChunks_ = traceFileReader_->NumChunks();
    }

    std::vector<Mongo::Data::TraceBatch<int16_t>> PopulateChunk()
    {
        std::vector<Mongo::Data::TraceBatch<int16_t>> chunk;
        Mongo::Data::BatchDimensions batchDims;
        batchDims.laneWidth = laneWidth_;
        batchDims.framesPerBatch = BlockLen();
        batchDims.lanesPerBatch = kernelLanes_;
        for (size_t b = 0; b < NumBatches(); b++)
        {
            // TODO: We allocate the memory here but do not de-allocate with the corresponding
            // call to DeactivateGpuMem().
            chunk.emplace_back(Mongo::Data::BatchMetadata(b, chunkIndex_ * BlockLen(),
                                                          (chunkIndex_ * BlockLen()) + BlockLen(),
                                                          b * batchDims.ZmwsPerBatch()),
                               batchDims,
                               Memory::SyncDirection::HostWriteDeviceRead,
                               SOURCE_MARKER());
            PopulateBatch(b, chunk.back());
        }
        chunkIndex_++;

        return chunk;
    }

    size_t NumBatches() const { return numZmwLanes_ / kernelLanes_; }

    bool Finished() const { return chunkIndex_ >= numChunks_; }

    size_t NumChunks() const { return numChunks_; }

    size_t NumZmwLanes() const { return numZmwLanes_; }

    size_t NumTraceChunks() const { return numTraceChunks_ ; }

    uint64_t NumFrames() const { return numChunks_ * BlockLen(); }

private:
    void PopulateBatch(uint32_t batchNum, Mongo::Data::TraceBatch<int16_t>& traceBatch)
    {
        if (!traceFileReader_) return;

        uint32_t traceStartZmwLane = (batchNum * kernelLanes_) % traceFileReader_->NumZmwLanes();
        uint32_t wrappedChunkIndex = chunkIndex_ % traceFileReader_->NumChunks();

        for (size_t lane = 0; lane < traceBatch.LanesPerBatch(); lane++)
        {
            auto block = traceBatch.GetBlockView(lane);
            uint32_t wrappedLane = (traceStartZmwLane + lane) % traceFileReader_->NumZmwLanes();
            PopulateBlockFromTraceFileSource(wrappedLane, wrappedChunkIndex, block.Data());
        }
    }

    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<int16_t>& v) override
    {
        PopulateBlockFromTraceFileSource(laneIdx, blockIdx, v.data());
    }

    void PopulateBlockFromTraceFileSource(size_t laneIdx,
                                          size_t blockIdx,
                                          int16_t* v)
    {
        if (traceFileReader_)
            traceFileReader_->PopulateBlock(laneIdx, blockIdx, v);
    }

private:
    uint32_t laneWidth_;
    uint32_t kernelLanes_;
    size_t numZmwLanes_;
    size_t numChunks_;
    size_t numTraceChunks_;
    size_t chunkIndex_;
    std::unique_ptr<TraceFileReader> traceFileReader_;
};

}}}

#endif
