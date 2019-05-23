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
    TraceFileReaderImpl(const std::string& traceFileName, uint32_t zmwsPerLane, uint32_t framesPerChunk, bool cache)
        : zmwsPerLane_(zmwsPerLane)
        , framesPerChunk_(framesPerChunk)
        , traceFile_(traceFileName)
        , numZmwLanes_(traceFile_.NUM_HOLES / zmwsPerLane)
        , numChunks_((traceFile_.NFRAMES + framesPerChunk - 1) / framesPerChunk)
        , cached_(cache)
    {
        if (cached_)
        {
            PBLOG_INFO << "Caching trace file";
            ReadEntireTraceFile();
        }
    }

    size_t NumChunks() const { return numChunks_; }

    size_t NumZmwLanes() const { return numZmwLanes_; }

    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       int16_t* v) const
    {
        FetchBlock(laneIdx, blockIdx, v);
    }

    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<int16_t>& v) const
    {
        size_t wrappedLane = laneIdx % numZmwLanes_;
        size_t wrappedBlock = blockIdx % numChunks_;

        FetchBlock(wrappedLane, wrappedBlock, v.data());
    }

    ~TraceFileReaderImpl() = default;

private:

    uint32_t FetchBlock(size_t laneIdx, size_t blockIdx, int16_t* data) const
    {
        if (cached_)
        {
            std::memcpy(data, pixelCache_[blockIdx][laneIdx].origin(),
                        framesPerChunk_ * zmwsPerLane_ * sizeof(int16_t));
            return framesPerChunk_;

        }
        else
        {
            return ReadZmwLaneBlock(laneIdx * zmwsPerLane_,
                                    zmwsPerLane_,
                                    blockIdx * framesPerChunk_,
                                    framesPerChunk_,
                                    data);
        }

    }

    void ReadEntireTraceFile()
    {
        PBLOG_INFO << "Reading trace file into memory...";
        pixelCache_.resize(boost::extents[numChunks_][numZmwLanes_][framesPerChunk_][zmwsPerLane_]);
        std::vector<int16_t> data(framesPerChunk_ * zmwsPerLane_);
        size_t lane;
        for (lane = 0; lane < numZmwLanes_; lane++)
        {
            for (size_t block = 0; block < numChunks_; block++)
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
    uint32_t zmwsPerLane_;
    uint32_t framesPerChunk_;
    SequelTraceFileHDF5 traceFile_;;
    size_t numZmwLanes_;
    size_t numChunks_;
    boost::multi_array<int16_t, 4> pixelCache_;
    bool cached_;
};

///////////////////////////////////////////////////////////////////////////////

TraceFileReader::~TraceFileReader() = default;

TraceFileReader::TraceFileReader(const std::string& traceFileName, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                                 bool cache)
    : pImpl_(std::make_unique<TraceFileReaderImpl>(traceFileName, zmwsPerLane, framesPerChunk, cache))
{}

size_t TraceFileReader::NumChunks() const
{
    return pImpl_->NumChunks();
}

size_t TraceFileReader::NumZmwLanes() const
{
    return pImpl_->NumZmwLanes();
}

void TraceFileReader::PopulateBlock(size_t laneIdx, size_t blockIdx, int16_t* v) const
{
    pImpl_->PopulateBlock(laneIdx, blockIdx, v);
}

void TraceFileReader::PopulateBlock(size_t laneIdx, size_t blockIdx, std::vector<int16_t>& v) const
{
    pImpl_->PopulateBlock(laneIdx, blockIdx, v);
}

}}}
