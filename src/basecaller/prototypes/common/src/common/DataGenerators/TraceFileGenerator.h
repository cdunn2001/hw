#ifndef TRACE_FILE_GENERATOR_H
#define TRACE_FILE_GENERATOR_H

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/GeneratorBase.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <dataTypes/TraceBatch.h>

#include <vector_types.h>

namespace PacBio {
namespace Cuda {
namespace Data {

using namespace PacBio::Mongo::Data;

struct TraceFileParams
{
    std::string traceFileName;
    size_t numTraceLanes = 0;

    TraceFileParams& TraceFileName(const std::string& v)    { traceFileName = v; return *this; }
    TraceFileParams& NumTraceLanes(size_t v)                { numTraceLanes = v; return *this; }
};

//
// Calling this class TraceFileGenerator is somewhat a misnomer
// as a trace file is not being generated but that the source
// of the data is the trace file. Data is read from the trace file
// and used to populate a given lane and block.
//

class TraceFileGenerator : public GeneratorBase<short2>
{
public:
    TraceFileGenerator(const DataManagerParams& params, const TraceFileParams& traceParams);

    TraceFileGenerator(const std::string& fileName,
                       uint32_t lanesPerPool, uint32_t zmwsPerLane, uint32_t framesPerChunk,
                       bool cache);

    ~TraceFileGenerator();

    size_t PopulateChunk(std::vector<TraceBatch<int16_t>>& chunk) const;

    unsigned int NumBatches() const;

    size_t NumChunks() const;

    size_t NumTraceChunks() const;

    size_t NumTraceZmwLanes() const;

    size_t NumReqZmwLanes() const;

    TraceFileGenerator& NumReqZmwLanes(uint32_t numZmwLanes);

    size_t NumFrames() const;

    TraceFileGenerator& NumFrames(uint32_t frames);

    size_t ChunkIndex() const;

    bool Finished() const;

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) override;

private:
    TraceFileParams traceParams_;
    class TraceFileGeneratorImpl;
    std::unique_ptr<TraceFileGeneratorImpl> pImpl_;
};

}}}

#endif
