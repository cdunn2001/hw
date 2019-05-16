#ifndef TRACE_FILE_READER_H
#define TRACE_FILE_READER_H

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <dataTypes/TraceBatch.h>

#include <vector_types.h>
namespace PacBio {
namespace Cuda {
namespace Data {

using namespace PacBio::Mongo::Data;

class TraceFileReader
{
public:
    TraceFileReader(const std::string& traceFileName, uint32_t zmwsPerLane, uint32_t framesPerChunk);

    TraceFileReader(const std::string& traceFileName,
                    uint32_t lanesPerPool, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                    bool cache);

    ~TraceFileReader();

public: // non-mutating methods
    size_t PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk) const;

    void PopulateBlock(size_t laneIdx, size_t blockIdx, std::vector<short2>& v) const;

    unsigned int NumBatches() const;

    size_t NumChunks() const;

    size_t NumTraceChunks() const;

    size_t NumTraceZmwLanes() const;

    size_t NumZmwLanes() const;

    size_t NumFrames() const;

    size_t ChunkIndex() const;

    bool Finished() const;

public: // mutating methods
    TraceFileReader& NumFrames(uint32_t frames);

    TraceFileReader& NumZmwLanes(uint32_t numZmwLanes);

private:
    class TraceFileReaderImpl;
    std::unique_ptr<TraceFileReaderImpl> pImpl_;
};

}}}
#endif
