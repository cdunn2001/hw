#include <boost/multi_array.hpp>

#include <common/DataGenerators/TraceFileReader.h>

#include <pacbio/logging/Logger.h>
#include <pacbio/primary/SequelTraceFile.h>

#include <vector_functions.h>

using PacBio::Primary::SequelTraceFileHDF5;

namespace PacBio {
namespace Cuda {
namespace Data {

class TraceFileReader::TraceFileReaderImpl
{
public:
    TraceFileReaderImpl(const std::string& traceFileName, uint32_t zmwsPerLane, uint32_t framesPerChunk)
        : zmwsPerLane_(zmwsPerLane)
        , framesPerChunk_(framesPerChunk)
        , traceFile_(traceFileName)
        , numTraceZmwLanes_(traceFile_.NUM_HOLES / zmwsPerLane)
        , numTraceChunks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , pixelCache_(boost::extents[numTraceChunks_][numTraceZmwLanes_][framesPerChunk_][zmwsPerLane/2])
    {
        PBLOG_INFO << "Total number of trace file lanes = " << numTraceZmwLanes_;
        PBLOG_INFO << "Total number of trace file blocks = " << numTraceChunks_;

        ReadEntireTraceFile();
    }

    TraceFileReaderImpl(const std::string& traceFileName,
                        uint32_t lanesPerPool, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                        bool cache)
        : lanesPerPool_(lanesPerPool)
        , zmwsPerLane_(zmwsPerLane)
        , framesPerChunk_(framesPerChunk)
        , traceFile_(traceFileName)
        , numTraceZmwLanes_(traceFile_.NUM_HOLES / zmwsPerLane) // This discards the last runt zmwLane.
        , numTraceChunks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , numZmwLanes_(numTraceZmwLanes_)
        , pixelCache_(boost::extents[numTraceChunks_][numTraceZmwLanes_][framesPerChunk][zmwsPerLane/2])
        , chunkIndex_(0)
        , numRequestedTraceChunks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , numRequestedBatches_((numTraceZmwLanes_ + lanesPerPool - 1) / lanesPerPool)
        , cached_(cache)
    {
        if (cached_)
        {
            PBLOG_INFO << "Caching trace file...";
            ReadEntireTraceFile();
        }
    }

    size_t PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk)
    {
        if (chunkIndex_ >= numRequestedTraceChunks_) return 0;

        std::vector<size_t> nFramesRead;
        nFramesRead.resize(NumBatches());
        for (size_t batchNum = 0; batchNum < NumBatches(); batchNum++)
            nFramesRead[batchNum] = PopulateTraceBatch(batchNum, chunk[batchNum]);

        if (std::all_of(nFramesRead.begin(), nFramesRead.end(),
                        [this](size_t nFramesRead) { return nFramesRead == this->framesPerChunk_; }))
        {
            chunkIndex_++;
            return framesPerChunk_;
        }
        else
        {
            PBLOG_ERROR << "Not all trace batches contain same number of frames = " << framesPerChunk_;
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
        return numZmwLanes_;
    }

    void NumZmwLanes(size_t numZmwLanes)
    {
        numZmwLanes_ = numZmwLanes;
        numRequestedBatches_ = (numZmwLanes_ + lanesPerPool_ - 1) / lanesPerPool_;
    }

    size_t Frames() const
    {
        return numRequestedTraceChunks_ * framesPerChunk_;
    }

    void Frames(uint32_t frames)
    {
        numRequestedTraceChunks_ = (frames + framesPerChunk_ - 1) / framesPerChunk_;
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
                  pixelCache_[wrappedBlock][wrappedLane].origin() + (framesPerChunk_ * (zmwsPerLane_/2)),
                  v.data());
    }

    ~TraceFileReaderImpl() = default;

private:
    size_t PopulateTraceBatch(uint32_t batchNum, TraceBatch<int16_t>& traceBatch) const
    {
        uint32_t traceStartZmwLane = (batchNum * lanesPerPool_) % numTraceZmwLanes_;
        uint32_t wrappedChunkIndex = chunkIndex_ % numTraceChunks_;

        for (size_t lane = 0; lane < traceBatch.LanesPerBatch(); lane++)
        {
            auto block = traceBatch.GetBlockView(lane);
            uint32_t wrappedLane = (traceStartZmwLane + lane) % numTraceZmwLanes_;

            if (cached_)
            {
                std::memcpy(block.Data(), pixelCache_[wrappedChunkIndex][wrappedLane].origin(),
                            framesPerChunk_ * zmwsPerLane_ * sizeof(int16_t));
            }
            else
            {
                ReadZmwLaneBlock(wrappedLane * zmwsPerLane_,
                                 zmwsPerLane_,
                                 wrappedChunkIndex * framesPerChunk_,
                                 framesPerChunk_,
                                 block.Data());
            }
        }

        return framesPerChunk_;
    }


    void ReadEntireTraceFile()
    {
        PBLOG_INFO << "Reading trace file into memory...";

        std::vector<int16_t> data(framesPerChunk_ * zmwsPerLane_);
        size_t lane;
        for (lane = 0; lane < numTraceZmwLanes_; lane++)
        {
            for (size_t block = 0; block < numTraceChunks_; block++)
            {
                ReadZmwLaneBlock(lane * zmwsPerLane_,
                                 zmwsPerLane_,
                                 block * framesPerChunk_,
                                 framesPerChunk_,
                                 data.data());

                std::memcpy(pixelCache_[block][lane].origin(), data.data(),
                            framesPerChunk_ * zmwsPerLane_ * sizeof(int16_t));
            }
        }
        PBLOG_INFO << "Done reading trace file into memory";
    }

    uint32_t ReadZmwLaneBlock(uint32_t zmwIndex, uint32_t numZmws, uint64_t frameOffset, uint32_t numFrames, int16_t* data) const
    {
        size_t nFramesRead = traceFile_.Read1CZmwLaneSegment(zmwIndex, numZmws, frameOffset, numFrames, data);
        if (nFramesRead < framesPerChunk_)
        {
            // Pad out if not enough frames.
            std::memset(data + (nFramesRead * zmwsPerLane_),
                        0, (framesPerChunk_ - nFramesRead) * zmwsPerLane_);
        }
        return framesPerChunk_;
    }

private:
    uint32_t lanesPerPool_;
    uint32_t zmwsPerLane_;
    uint32_t framesPerChunk_;
    SequelTraceFileHDF5 traceFile_;
    size_t numTraceZmwLanes_;
    size_t numTraceChunks_;
    size_t numZmwLanes_;
    boost::multi_array<short2, 4> pixelCache_;
    size_t chunkIndex_;
    size_t numRequestedTraceChunks_;
    size_t numRequestedBatches_;
    bool cached_;
};

///////////////////////////////////////////////////////////////////////////////

TraceFileReader::~TraceFileReader() = default;

TraceFileReader::TraceFileReader(const std::string& traceFileName, uint32_t zmwsPerLane, uint32_t framesPerChunk)
    : pImpl_(std::make_unique<TraceFileReaderImpl>(traceFileName, zmwsPerLane, framesPerChunk))
{}

TraceFileReader::TraceFileReader(const std::string& traceFileName,
                                 uint32_t lanesPerPool, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                                 bool cache)
    : pImpl_(std::make_unique<TraceFileReaderImpl>(traceFileName,
                                                   lanesPerPool, zmwsPerLane, framesPerChunk,
                                                   cache))
{}

size_t TraceFileReader::PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk) const
{
    return pImpl_->PopulateChunk(chunk);
}

unsigned int TraceFileReader::NumBatches() const
{
    return pImpl_->NumBatches();
}

size_t TraceFileReader::NumChunks() const
{
    return pImpl_->NumChunks();
}

size_t TraceFileReader::NumTraceChunks() const
{
    return pImpl_->NumTraceChunks();
}

size_t TraceFileReader::NumTraceZmwLanes() const
{
    return pImpl_->NumTraceZmwLanes();
}

size_t TraceFileReader::NumZmwLanes() const
{
    return pImpl_->NumZmwLanes();
}

TraceFileReader& TraceFileReader::NumZmwLanes(uint32_t numZmwLanes)
{
    pImpl_->NumZmwLanes(numZmwLanes);
    return *this;
}

size_t TraceFileReader::NumFrames() const
{
    return pImpl_->Frames();
}

TraceFileReader& TraceFileReader::NumFrames(uint32_t frames)
{
    pImpl_->Frames(frames);
    return *this;
}

size_t TraceFileReader::ChunkIndex() const
{
    return pImpl_->ChunkIndex();
}

bool TraceFileReader::Finished() const
{
    return pImpl_->Finished();
}

void TraceFileReader::PopulateBlock(size_t laneIdx, size_t blockIdx, std::vector<short2>& v) const
{
    pImpl_->PopulateBlock(laneIdx, blockIdx, v);
}

}}}