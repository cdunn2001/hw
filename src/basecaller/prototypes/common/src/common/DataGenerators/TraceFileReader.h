#ifndef TRACE_FILE_READER_H
#define TRACE_FILE_READER_H

#include <vector_types.h>

namespace PacBio {
namespace Cuda {
namespace Data {

class TraceFileReader
{
public:
    TraceFileReader(const std::string& traceFileName, uint32_t zmwsPerLane, uint32_t framesPerChunk, bool cache=true);

    ~TraceFileReader();

public: // non-mutating methods
    void PopulateBlock(size_t laneIdx, size_t blockIdx, int16_t* v) const;
    void PopulateBlock(size_t laneIdx, size_t blockIdx, std::vector<short2>& v) const;

    size_t NumChunks() const;

    size_t NumZmwLanes() const;

private:
    class TraceFileReaderImpl;
    std::unique_ptr<TraceFileReaderImpl> pImpl_;
};

}}}
#endif
