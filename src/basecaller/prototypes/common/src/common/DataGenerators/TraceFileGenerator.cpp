
#include <boost/multi_array.hpp>

#include <common/DataGenerators/TraceFileGenerator.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/primary/SequelTraceFile.h>

#include <vector_functions.h>

using PacBio::Primary::SequelTraceFileHDF5;

namespace PacBio {
namespace Cuda {
namespace Data {

class TraceFileGenerator::TraceFileGeneratorImpl
{
public:
    TraceFileGeneratorImpl(const DataManagerParams& dataParams, const TraceFileParams& traceParams)
        : dataParams_(dataParams)
        , traceParams_(traceParams)
        , traceFile_(traceParams_.traceFileName)
        , numTraceZmwLanes_(traceParams_.numTraceLanes == 0 ? traceFile_.NUM_HOLES / (dataParams_.zmwLaneWidth) : traceParams_.numTraceLanes)
        , numTraceChunks_((traceFile_.NFRAMES + dataParams_.blockLength - 1) / dataParams_.blockLength)
        , pixelCache_(boost::extents[numTraceChunks_][numTraceZmwLanes_][dataParams_.blockLength][dataParams_.gpuLaneWidth])
    {
        PBLOG_INFO << "Total number of trace file lanes = " << numTraceZmwLanes_;
        PBLOG_INFO << "Total number of trace file blocks = " << numTraceChunks_;

        ReadEntireTraceFile();
    }

    TraceFileGeneratorImpl(const std::string& fileName,
                           uint32_t lanesPerPool, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                           bool cache)
        : traceFile_(fileName)
        , numTraceZmwLanes_(traceFile_.NUM_HOLES / zmwsPerLane) // This discards the last runt zmwLane.
        , numTraceChunks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , pixelCache_(boost::extents[numTraceChunks_][numTraceZmwLanes_][framesPerChunk][zmwsPerLane/2])
        , chunkIndex_(0)
        , numRequestedTraceChunks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , numRequestedBatches_((numTraceZmwLanes_ + lanesPerPool - 1) / lanesPerPool)
        , cached_(cache)
    {
        dataParams_ = DataManagerParams()
                .ZmwLaneWidth(zmwsPerLane)
                .NumZmwLanes(numTraceZmwLanes_)
                .KernelLanes(lanesPerPool)
                .BlockLength(framesPerChunk);

        traceParams_ = TraceFileParams()
                .TraceFileName(fileName);

        if (cached_)
        {
            PBLOG_INFO << "Caching trace file...";
            ReadEntireTraceFile();
        }
    }

    size_t PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk)
    {
        if (chunkIndex_ >= numRequestedTraceChunks_) return 0;

        BatchDimensions batchDims;
        batchDims.laneWidth = dataParams_.zmwLaneWidth;
        batchDims.framesPerBatch = dataParams_.blockLength;
        batchDims.lanesPerBatch = dataParams_.kernelLanes;

        std::vector<size_t> nFramesRead;
        nFramesRead.resize(NumBatches());
        for (size_t batchNum = 0; batchNum < NumBatches(); batchNum++)
        {
            BatchMetadata batchMetadata(batchNum,
                                        chunkIndex_ * dataParams_.blockLength,
                                        (chunkIndex_ * dataParams_.blockLength) + dataParams_.blockLength);
            chunk.emplace_back(batchMetadata, batchDims, Cuda::Memory::SyncDirection::Symmetric);
            nFramesRead[batchNum] = PopulateTraceBatch(batchNum, chunk.back());
        }

        if (std::all_of(nFramesRead.begin(), nFramesRead.end(),
                        [this](size_t nFramesRead) { return nFramesRead == this->dataParams_.blockLength; }))
        {
            chunkIndex_++;
            return dataParams_.blockLength;
        }
        else
        {
            PBLOG_ERROR << "Not all trace batches contain same number of frames = " << dataParams_.blockLength;
            return 0;
        }
    }

    unsigned int NumBatches() const
    {
        return numRequestedBatches_;
    }

    size_t NumChunks() const
    {
        return numRequestedTraceChunks_;
    }
    
    size_t NumTraceChunks() const
    {
        return numTraceChunks_;
    }
    
    size_t NumTraceZmwLanes() const
    {
        return numTraceZmwLanes_;
    }
    
    size_t NumZmwLanes() const
    {
        return dataParams_.numZmwLanes;
    }

    void NumZmwLanes(size_t numZmwLanes)
    {
        dataParams_.numZmwLanes = numZmwLanes;
        numRequestedBatches_ = (numZmwLanes + dataParams_.kernelLanes - 1) / dataParams_.kernelLanes;
    }

    size_t Frames() const
    {
        return numRequestedTraceChunks_ * dataParams_.blockLength;
    }

    void Frames(uint32_t frames)
    {
        numRequestedTraceChunks_ = (frames + dataParams_.blockLength - 1) / dataParams_.blockLength;
    }

    size_t ChunkIndex() const
    {
        return chunkIndex_;
    }

    bool Finished() const
    {
        return chunkIndex_ >= numRequestedTraceChunks_;
    }

    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) const
    {
        size_t wrappedLane = laneIdx % numTraceZmwLanes_;
        size_t wrappedBlock = blockIdx % numTraceChunks_;

        std::copy(pixelCache_[wrappedBlock][wrappedLane].origin(),
                  pixelCache_[wrappedBlock][wrappedLane].origin() + (dataParams_.blockLength * dataParams_.gpuLaneWidth),
                  v.data());
    }

    ~TraceFileGeneratorImpl() = default;

private:
    size_t PopulateTraceBatch(uint32_t batchNum, TraceBatch<int16_t>& traceBatch) const
    {
        uint32_t traceStartZmwLane = (batchNum * dataParams_.kernelLanes) % numTraceZmwLanes_;
        uint32_t wrappedChunkIndex = chunkIndex_ % numTraceChunks_;

        for (size_t lane = 0; lane < traceBatch.LanesPerBatch(); lane++)
        {
            auto block = traceBatch.GetBlockView(lane);
            uint32_t wrappedLane = (traceStartZmwLane + lane) % numTraceZmwLanes_;

            if (cached_)
            {
                std::memcpy(block.Data(), pixelCache_[wrappedChunkIndex][wrappedLane].origin(),
                            dataParams_.blockLength * dataParams_.zmwLaneWidth * sizeof(int16_t));
            }
            else
            {
                ReadZmwLaneBlock(wrappedLane * dataParams_.zmwLaneWidth,
                                 dataParams_.zmwLaneWidth,
                                 wrappedChunkIndex * dataParams_.blockLength,
                                 dataParams_.blockLength,
                                 block.Data());
            }
        }

        return dataParams_.blockLength;
    }


    void ReadEntireTraceFile()
    {
        PBLOG_INFO << "Reading trace file into memory...";

        std::vector<int16_t> data(dataParams_.blockLength * dataParams_.zmwLaneWidth);
        size_t lane;
        for (lane = 0; lane < numTraceZmwLanes_; lane++)
        {
            for (size_t block = 0; block < numTraceChunks_; block++)
            {
                ReadZmwLaneBlock(lane * dataParams_.zmwLaneWidth,
                                 dataParams_.zmwLaneWidth,
                                 block * dataParams_.blockLength,
                                 dataParams_.blockLength,
                                 data.data());

                std::memcpy(pixelCache_[block][lane].origin(), data.data(),
                            dataParams_.blockLength * dataParams_.zmwLaneWidth * sizeof(int16_t));
            }
        }
        PBLOG_INFO << "Done reading trace file into memory";
    }

    uint32_t ReadZmwLaneBlock(uint32_t zmwIndex, uint32_t numZmws, uint64_t frameOffset, uint32_t numFrames, int16_t* data) const
    {
        size_t nFramesRead = traceFile_.Read1CZmwLaneSegment(zmwIndex, numZmws, frameOffset, numFrames, data);
        if (nFramesRead < dataParams_.blockLength)
        {
            // Pad out if not enough frames.
            std::memset(data + (nFramesRead * dataParams_.zmwLaneWidth),
                        0, (dataParams_.blockLength - nFramesRead) * dataParams_.zmwLaneWidth);
        }
        return dataParams_.blockLength;
    }

private:
    DataManagerParams dataParams_;
    TraceFileParams traceParams_;
    SequelTraceFileHDF5 traceFile_;
    size_t numTraceZmwLanes_;
    size_t numTraceChunks_;
    boost::multi_array<short2, 4> pixelCache_;
    size_t chunkIndex_;
    size_t numRequestedTraceChunks_;
    size_t numRequestedBatches_;
    bool cached_;
};

///////////////////////////////////////////////////////////////////////////////

TraceFileGenerator::~TraceFileGenerator() = default;

TraceFileGenerator::TraceFileGenerator(const DataManagerParams& params, const TraceFileParams& traceParams)
    : GeneratorBase(params.blockLength,
                    params.gpuLaneWidth,
                    params.numBlocks,
                    params.numZmwLanes)
    , traceParams_(traceParams)
    , pImpl_(std::make_unique<TraceFileGeneratorImpl>(params, traceParams))
    {}

TraceFileGenerator::TraceFileGenerator(const std::string& fileName,
                                       uint32_t lanesPerPool, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                                       bool cache)
    : GeneratorBase(framesPerChunk,
                    zmwsPerLane/2,
                    0,
                    0)
    , traceParams_(TraceFileParams().TraceFileName(fileName))
    , pImpl_(std::make_unique<TraceFileGeneratorImpl>(fileName,
                                                      lanesPerPool, zmwsPerLane, framesPerChunk,
                                                      cache))
    {}

size_t TraceFileGenerator::PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk) const
{
    return pImpl_->PopulateChunk(chunk);
}

unsigned int TraceFileGenerator::NumBatches() const
{
    return pImpl_->NumBatches();
}

size_t TraceFileGenerator::NumChunks() const
{
    return pImpl_->NumChunks();
}

size_t TraceFileGenerator::NumTraceChunks() const
{
    return pImpl_->NumTraceChunks();
}

size_t TraceFileGenerator::NumTraceZmwLanes() const
{
    return pImpl_->NumTraceZmwLanes();
}

size_t TraceFileGenerator::NumReqZmwLanes() const
{
    return pImpl_->NumZmwLanes();
}

TraceFileGenerator& TraceFileGenerator::NumReqZmwLanes(uint32_t numZmwLanes)
{
    pImpl_->NumZmwLanes(numZmwLanes);
    return *this;
}

size_t TraceFileGenerator::NumFrames() const
{
    return pImpl_->Frames();
}

TraceFileGenerator& TraceFileGenerator::NumFrames(uint32_t frames)
{
    pImpl_->Frames(frames);
    return *this;
}

size_t TraceFileGenerator::ChunkIndex() const
{
    return pImpl_->ChunkIndex();
}

bool TraceFileGenerator::Finished() const
{
    return pImpl_->Finished();
}

void TraceFileGenerator::PopulateBlock(size_t laneIdx,
                                       size_t blockIdx,
                                       std::vector<short2>& v)
{
    pImpl_->PopulateBlock(laneIdx, blockIdx, v);
}

}}}