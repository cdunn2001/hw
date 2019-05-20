#ifndef BATCH_GENERATOR_H
#define BATCH_GENERATOR_H

#include <common/DataGenerators/GeneratorBase.h>
#include <common/DataGenerators/TraceFileReader.h>
#include <dataTypes/BatchMetadata.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Cuda {
namespace Data {

class BatchGenerator : public GeneratorBase<int16_t>
{
public:
    BatchGenerator(uint32_t blockLen, uint32_t zmwLaneWidth, uint32_t kernelLanes,
                   uint32_t frames, uint32_t numZmwLanes)
        : GeneratorBase(blockLen,
                        zmwLaneWidth/2,
                        (frames + blockLen - 1)/blockLen,
                        numZmwLanes)
        , zmwLaneWidth_(zmwLaneWidth)
        , kernelLanes_(kernelLanes)
        , numZmwLanes_(numZmwLanes)
        , numChunks_(NumBlocks())
        , numTraceChunks_(0)
        , chunkIndex_(0)
    {
        tracePool_ = std::make_shared<Memory::DualAllocationPools>(zmwLaneWidth * blockLen * kernelLanes * sizeof(int16_t));
    }

    void SetTraceFileSource(const std::string& traceFileName, bool cache)
    {
        traceFileReader_.reset(new TraceFileReader(traceFileName, zmwLaneWidth_, BlockLen(), cache));
        if (numZmwLanes_ == 0) numZmwLanes_ = traceFileReader_->NumZmwLanes();
        if (numChunks_ == 0) numChunks_ = traceFileReader_->NumChunks();
        numTraceChunks_ = traceFileReader_->NumChunks();
    }

    std::vector<Mongo::Data::TraceBatch<int16_t>> PopulateChunk()
    {
        std::vector<Mongo::Data::TraceBatch<int16_t>> chunk;
        Mongo::Data::BatchDimensions batchDims;
        batchDims.laneWidth = zmwLaneWidth_;
        batchDims.framesPerBatch = BlockLen();
        batchDims.lanesPerBatch = kernelLanes_;
        for (size_t b = 0; b < NumBatches(); b++)
        {
            // TODO: We allocate the memory here but do not de-allocate with the corresponding
            // call to DeactivateGpuMem().
            chunk.emplace_back(Mongo::Data::BatchMetadata(b, chunkIndex_ * BlockLen(),
                                                          (chunkIndex_ * BlockLen()) + BlockLen()),
                               batchDims,
                               Memory::SyncDirection::HostWriteDeviceRead,
                               tracePool_);
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

private:
    void PopulateBatch(uint32_t batchNum, Mongo::Data::TraceBatch<int16_t>& traceBatch)
    {
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
        traceFileReader_->PopulateBlock(laneIdx, blockIdx, v);
    }

private:
    uint32_t zmwLaneWidth_;
    uint32_t kernelLanes_;
    size_t numZmwLanes_;
    size_t numChunks_;
    size_t numTraceChunks_;
    size_t chunkIndex_;
    std::shared_ptr<Memory::DualAllocationPools> tracePool_;
    std::unique_ptr<TraceFileReader> traceFileReader_;
};

}}}

#endif
