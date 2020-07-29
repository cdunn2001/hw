#ifndef TRACE_FILE_READER_H
#define TRACE_FILE_READER_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <vector_types.h>

#include <pacbio/tracefile/TraceFile.h>
#include <pacbio/PBException.h>

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

    size_t NumChunks() const
    { return numChunks_; }

    size_t NumZmwLanes() const
    { return numZmwLanes_; }

private:
    void ReadEntireTraceFile();

    uint32_t FetchBlock(size_t laneIdx, size_t blockIdx, int16_t* data)
    {
        if (laneIdx >= numZmwLanes_) throw PBException("Bad lane index given!");
        if (blockIdx >= numChunks_) throw PBException("Bad chunk index given!");

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
        size_t nZmwsToRead = std::min(numZmws, numZmws_ - zmwIndex);
        std::vector<int16_t> d(nZmwsToRead*nFramesToRead);
        using range = boost::multi_array_types::extent_range;
        const range zmwRange(zmwIndex, zmwIndex+nZmwsToRead);
        const range frameRange(frameOffset, frameOffset+nFramesToRead);
        boost::multi_array_ref<int16_t,2> dataIn{d.data(), boost::extents[zmwRange][frameRange]};
        traceFile_.Traces().ReadTraceBlock(dataIn);
        for (size_t frame = 0; frame < nFramesToRead; frame++)
        {
            for(size_t zmw = 0; zmw < nZmwsToRead; zmw++)
            {
                data[(frame*nZmwsToRead)+zmw] = d[(zmw*nFramesToRead)+frame];
            }
        }
        if (nZmwsToRead*nFramesToRead < zmwsPerLane_*framesPerChunk_)
        {
            // Pad out if not enough frames.
            std::memset(data + (nFramesToRead * nZmwsToRead),
                        0, (framesPerChunk_ - nFramesToRead) * (zmwsPerLane_ - nZmwsToRead));
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
