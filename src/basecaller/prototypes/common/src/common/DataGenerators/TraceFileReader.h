#ifndef TRACE_FILE_READER_H
#define TRACE_FILE_READER_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <vector_types.h>

#include <pacbio/tracefile/TraceFile.h>

namespace PacBio {
namespace Cuda {
namespace Data {

class TraceFileReader
{
public:
    TraceFileReader(const std::string& traceFileName, size_t zmwsPerLane,
                    size_t framesPerChunk, bool cache=true);

public:
    void PopulateBlock(size_t laneIdx, size_t blockIdx, int16_t* v)
    {
        FetchBlock(laneIdx, blockIdx, v);
    }

    void PopulateBlock(size_t laneIdx, size_t blockIdx, std::vector<int16_t>& v)
    {
        size_t wrappedLane = laneIdx % numZmwLanes_;
        size_t wrappedBlock = blockIdx % numChunks_;
        FetchBlock(wrappedLane, wrappedBlock, v.data());
    }

    size_t NumChunks() const
    { return numChunks_; }

    size_t NumZmwLanes() const
    { return numZmwLanes_; }

private:
    void ReadEntireTraceFile();

    uint32_t FetchBlock(size_t laneIdx, size_t blockIdx, int16_t* data)
    {
        if (cached_)
        {
            std::memcpy(data, pixelCache_[blockIdx][laneIdx].origin(),
                        framesPerChunk_ * zmwsPerLane_ * sizeof(int16_t));
            return framesPerChunk_;
        }
        else
        {
            if (laneCurrentChunk_[laneIdx] < blockIdx)
            {
                ReadZmwLaneBlock(laneIdx * zmwsPerLane_,
                                 zmwsPerLane_,
                                 blockIdx * framesPerChunk_,
                                 framesPerChunk_,
                                 pixelCache_[0][laneIdx].origin());
                laneCurrentChunk_[laneIdx] = (laneCurrentChunk_[laneIdx] + 1) % numChunks_;
            }
            std::memcpy(data, pixelCache_[0][laneIdx].origin(),
                        framesPerChunk_ * zmwsPerLane_ * sizeof(int16_t));
            return framesPerChunk_;
        }
    }

    uint32_t ReadZmwLaneBlock(size_t zmwIndex, size_t numZmws, size_t frameOffset,
                              size_t numFrames, int16_t* data) const
    {
        size_t nFramesToRead = std::min(numFrames, numFrames_ - frameOffset);
        std::vector<int16_t> d(numZmws*nFramesToRead);
        boost::multi_array_ref<int16_t,2> dataIn{d.data(), boost::extents[numZmws][nFramesToRead]};
        dataIn.reindex(boost::array<boost::multi_array<int16_t,2>::index,2>{zmwIndex, frameOffset});
        traceFile_.Traces().ReadTraceBlock(dataIn);
        for (size_t frame = 0; frame < nFramesToRead; frame++)
        {
            for(size_t zmw = 0; zmw < numZmws; zmw++)
            {
                data[(frame*numZmws)+zmw] = d[(zmw*nFramesToRead) + frame];
            }
        }
        if (nFramesToRead < framesPerChunk_)
        {
            // Pad out if not enough frames.
            std::memset(data + (nFramesToRead * zmwsPerLane_),
                        0, (framesPerChunk_ - nFramesToRead) * zmwsPerLane_);
        }
        return framesPerChunk_;
    }

private:
    TraceFile::TraceFile traceFile_;
    size_t zmwsPerLane_;
    size_t framesPerChunk_;
    size_t numFrames_;
    size_t numZmws_;
    size_t numZmwLanes_;
    size_t numChunks_;
    bool cached_;
    boost::multi_array<int16_t, 4> pixelCache_;
    std::vector<size_t> laneCurrentChunk_;
};

}}}

#endif /* TRACE_FILE_READER_H */
