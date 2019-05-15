
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
        , numTraceBlocks_((traceFile_.NFRAMES + dataParams_.blockLength - 1) / dataParams_.blockLength)
        , pixelCache_(boost::extents[numTraceBlocks_][numTraceZmwLanes_][dataParams_.blockLength][dataParams_.gpuLaneWidth])
    {
        PBLOG_INFO << "Total number of trace file lanes = " << numTraceZmwLanes_;
        PBLOG_INFO << "Total number of trace file blocks = " << numTraceBlocks_;

        ReadEntireTraceFile();
    }

    TraceFileGeneratorImpl(const std::string& fileName,
                           uint32_t zmwsPerLane, uint32_t lanesPerPool, uint32_t framesPerChunk,
                           uint32_t tileBatches, uint32_t tileChunks, bool cache)
        : traceFile_(fileName)
        , numTraceZmwLanes_(traceFile_.NUM_HOLES / zmwsPerLane)
        , numTraceBlocks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , chunkIndex_(0)
        , cached_(cache)
    {
        dataParams_ = DataManagerParams()
                .ZmwLaneWidth(zmwsPerLane)
                .NumZmwLanes(traceFile_.NUM_HOLES / zmwsPerLane)
                .KernelLanes(lanesPerPool)
                .BlockLength(framesPerChunk);

        traceParams_ = TraceFileParams()
                .TraceFileName(fileName);

        if (dataParams_.numZmwLanes % dataParams_.kernelLanes != 0)
            throw PBException("Number of zmws = " + std::to_string(traceFile_.NUM_HOLES) +
                              " in trace file must be multiple of zmwsPerLane x lanesPerPool = "
                              + std::to_string(zmwsPerLane * lanesPerPool));

        if (traceFile_.NFRAMES % dataParams_.blockLength != 0)
            throw PBException("Number of frames = " + std::to_string(traceFile_.NFRAMES) +
                              " in trace file must be multiple of framesPerChunk = " + std::to_string(framesPerChunk));
        if (tileChunks != 0)
        {
            if (tileChunks < numTraceBlocks_)
                throw PBException("Requested chunks to tile = " + std::to_string(tileChunks) +
                                  " must be greater than number of chunks in trace file = " +
                                  std::to_string(numTraceBlocks_));
            numRequestedTraceBlocks_ = tileChunks;
        }
        else
            numRequestedTraceBlocks_ = numTraceBlocks_;

        if (tileBatches != 0)
        {
            if (tileBatches < (dataParams_.numZmwLanes / dataParams_.kernelLanes))
                throw PBException("Requested batches to tile = " + std::to_string(tileBatches) +
                                  " must be greater than number of batches in trace file = " +
                                  std::to_string(dataParams_.numZmwLanes / dataParams_.kernelLanes));
            numRequestedBatches_ = tileBatches;
        }
        else
            numRequestedBatches_ = dataParams_.numZmwLanes / dataParams_.kernelLanes;

        if (cached_)
        {
            CacheTraceFile();
        }
    }

    size_t PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk)
    {
        if (chunkIndex_ >= numRequestedTraceBlocks_) return 0;

        BatchDimensions batchDims;
        batchDims.laneWidth = dataParams_.zmwLaneWidth;
        batchDims.framesPerBatch = dataParams_.blockLength;
        batchDims.lanesPerBatch = dataParams_.kernelLanes;

        std::vector<size_t> nFramesRead;
        nFramesRead.resize(GetNumBatches());
        for (size_t batchNum = 0; batchNum < GetNumBatches(); batchNum++)
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
            return 0;
    }

    unsigned int GetNumBatches() const
    {
        return numRequestedBatches_;
    }

    size_t GetNumChunks() const
    {
        return numRequestedTraceBlocks_;
    }

    bool Finished() const
    {
        return chunkIndex_ >= numRequestedTraceBlocks_;
    }

    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) const
    {
        size_t wrappedLane = laneIdx % numTraceZmwLanes_;
        size_t wrappedBlock = blockIdx % numTraceBlocks_;

        std::copy(pixelCache_[wrappedBlock][wrappedLane].origin(),
                  pixelCache_[wrappedBlock][wrappedLane].origin() + (dataParams_.blockLength * dataParams_.gpuLaneWidth),
                  v.data());
    }

    ~TraceFileGeneratorImpl() = default;

private:
    void CacheTraceFile()
    {
        PBLOG_INFO << "Caching trace file...";
        pixelCache_.resize(boost::extents[numTraceBlocks_][numTraceZmwLanes_][dataParams_.blockLength][dataParams_.gpuLaneWidth]);
        ReadEntireTraceFile();
    }

    size_t PopulateTraceBatch(uint32_t batchNum, TraceBatch<int16_t>& traceBatch) const
    {
        uint32_t wrappedBatchNum = batchNum % (dataParams_.numZmwLanes / dataParams_.kernelLanes);
        uint32_t wrappedChunkIndex = chunkIndex_ % numTraceBlocks_;

        if (cached_)
        {
            size_t startLane = wrappedBatchNum * dataParams_.kernelLanes;
            for (size_t lane = 0; lane < traceBatch.LanesPerBatch(); lane++)
            {
                auto block = traceBatch.GetBlockView(lane);
                std::memcpy(block.Data(), pixelCache_[wrappedChunkIndex][startLane + lane].origin(),
                            dataParams_.blockLength * dataParams_.zmwLaneWidth * sizeof(int16_t));
            }
            return dataParams_.blockLength;
        }
        else
        {
            std::vector<int16_t> data(dataParams_.blockLength * dataParams_.zmwLaneWidth * dataParams_.kernelLanes);

            size_t nFramesRead = ReadZmwLaneBlock(
                    wrappedBatchNum * (dataParams_.zmwLaneWidth * dataParams_.kernelLanes),
                    (dataParams_.zmwLaneWidth * dataParams_.kernelLanes),
                    wrappedChunkIndex * dataParams_.blockLength,
                    dataParams_.blockLength,
                    data.data());

            for (size_t lane = 0; lane < traceBatch.LanesPerBatch(); lane++)
            {
                auto block = traceBatch.GetBlockView(lane);
                std::memcpy(block.Data(), data.data() + (lane * block.LaneWidth()), block.LaneWidth());
            }

            return nFramesRead;
        }
    }


    void ReadEntireTraceFile()
    {
        PBLOG_INFO << "Reading trace file into memory...";

        std::vector<int16_t> data(dataParams_.blockLength * dataParams_.zmwLaneWidth);
        size_t lane;
        for (lane = 0; lane < numTraceZmwLanes_; lane++)
        {
            for (size_t block = 0; block < numTraceBlocks_; block++)
            {
                ReadZmwLaneBlock(lane * dataParams_.zmwLaneWidth,
                                 dataParams_.zmwLaneWidth,
                                 block * dataParams_.blockLength,
                                 dataParams_.blockLength,
                                 data.data());

                std::memcpy(pixelCache_[block][lane].origin(), data.data(),
                            dataParams_.blockLength * dataParams_.zmwLaneWidth * sizeof(int16_t));
            }

            if (lane % 10000 == 0) PBLOG_INFO << "Read " << lane << "/" << numTraceZmwLanes_;
        }
        PBLOG_INFO << "Read " << lane << "/" << numTraceZmwLanes_ << "...Done reading trace file into memory";
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
        return nFramesRead;
    }

private:
    DataManagerParams dataParams_;
    TraceFileParams traceParams_;
    SequelTraceFileHDF5 traceFile_;
    size_t numTraceZmwLanes_;
    size_t numTraceBlocks_;
    boost::multi_array<short2, 4> pixelCache_;
    size_t chunkIndex_;
    size_t numRequestedTraceBlocks_;
    size_t numRequestedBatches_;
    bool cached_;
};

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
                                       uint32_t zmwsPerLane, uint32_t lanesPerPool, uint32_t framesPerChunk,
                                       uint32_t tileBatches, uint32_t tileChunks, bool cache)
    : GeneratorBase(framesPerChunk,
                    zmwsPerLane/2,
                    0, 0)
    , pImpl_(std::make_unique<TraceFileGeneratorImpl>(fileName, zmwsPerLane, lanesPerPool, framesPerChunk,
                                                      tileBatches, tileChunks, cache))
    {}

size_t TraceFileGenerator::PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk) const
{
    return pImpl_->PopulateChunk(chunk);
}

unsigned int TraceFileGenerator::GetNumBatches() const
{
    return pImpl_->GetNumBatches();
}

size_t TraceFileGenerator::GetNumChunks() const
{
    return pImpl_->GetNumChunks();
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