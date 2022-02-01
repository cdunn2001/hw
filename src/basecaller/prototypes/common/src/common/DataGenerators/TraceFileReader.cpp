#include <boost/multi_array.hpp>

#include <common/DataGenerators/TraceFileReader.h>

#include <pacbio/logging/Logger.h>

#include <vector_functions.h>


namespace PacBio {
namespace Cuda {
namespace Data {

TraceFileReader::TraceFileReader(const std::string& traceFileName, size_t zmwsPerLane,
                                 size_t framesPerChunk, bool cache)
    : traceFile_(traceFileName)
    , zmwsPerLane_(zmwsPerLane)
    , framesPerChunk_(framesPerChunk)
    , numFrames_(traceFile_.Traces().NumFrames())
    , numZmws_(traceFile_.Traces().NumZmws())
    , numZmwLanes_(numZmws_ / zmwsPerLane_)
    , numChunks_(numFrames_ / framesPerChunk_)
    , cached_(cache)
{
    if (cached_)
    {
        PBLOG_INFO << "Caching trace file";
        ReadEntireTraceFile();
    }
    else
    {
        pixelCache_.resize(boost::extents[1][numZmwLanes_][framesPerChunk_][zmwsPerLane_]);
        laneCurrentChunk_.resize(numZmwLanes_, std::numeric_limits<size_t>::max());
    }
}

void TraceFileReader::ReadEntireTraceFile()
{
    PBLOG_INFO << "Reading trace file into memory...";
    pixelCache_.resize(boost::extents[numChunks_][numZmwLanes_][framesPerChunk_][zmwsPerLane_]);
    for (size_t lane = 0; lane < numZmwLanes_; lane++)
    {
        for (size_t block = 0; block < numChunks_; block++)
        {
            ReadZmwLaneBlock(lane * zmwsPerLane_,
                             zmwsPerLane_,
                             block * framesPerChunk_,
                             framesPerChunk_,
                             pixelCache_[block][lane].origin());
        }
    }
    PBLOG_INFO << "Done reading trace file into memory";
}

}}}
