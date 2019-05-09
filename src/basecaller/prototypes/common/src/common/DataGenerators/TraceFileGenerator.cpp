
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

    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v)
    {
        size_t wrappedLane = laneIdx % numTraceZmwLanes_;
        size_t wrappedBlock = blockIdx % numTraceBlocks_;

        std::copy(pixelCache_[wrappedBlock][wrappedLane].origin(),
                  pixelCache_[wrappedBlock][wrappedLane].origin() + (dataParams_.blockLength * dataParams_.gpuLaneWidth),
                  v.data());
    }

    ~TraceFileGeneratorImpl() = default;

private:
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

    void ReadZmwLaneBlock(uint32_t zmwIndex, uint32_t numZmws, uint64_t frameOffset, uint32_t numFrames, int16_t* data)
    {
        size_t nFramesRead = traceFile_.Read1CZmwLaneSegment(zmwIndex, numZmws, frameOffset, numFrames, data);
        if (nFramesRead < dataParams_.blockLength)
        {
            // Pad out if not enough frames.
            std::memset(data + (nFramesRead * dataParams_.zmwLaneWidth),
                        0, (dataParams_.blockLength - nFramesRead) * dataParams_.zmwLaneWidth);
        }

    }

private:
    DataManagerParams dataParams_; 
    TraceFileParams traceParams_;
    SequelTraceFileHDF5 traceFile_;
    size_t numTraceZmwLanes_;
    size_t numTraceBlocks_;
    boost::multi_array<short2, 4> pixelCache_;
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

void TraceFileGenerator::PopulateBlock(size_t laneIdx,
                                       size_t blockIdx,
                                       std::vector<short2>& v)
{
    pImpl_->PopulateBlock(laneIdx, blockIdx, v);
}

}}}