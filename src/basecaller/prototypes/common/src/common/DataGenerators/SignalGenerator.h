#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/GeneratorBase.h>
#include <common/DataGenerators/TraceFileReader.h>
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

class SignalGenerator : public GeneratorBase<short2>
{
public:
    SignalGenerator(const DataManagerParams& params, const TraceFileParams& traceParams)
        : GeneratorBase(params.blockLength,
                        params.gpuLaneWidth,
                        params.numBlocks,
                        params.numZmwLanes)
        , traceParams_(traceParams)
        , inputTraceFile_(std::make_unique<TraceFileReader>(traceParams_.traceFileName, params.zmwLaneWidth, params.blockLength))
    {}

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) override
    {
        inputTraceFile_->PopulateBlock(laneIdx, blockIdx, v);
    }

private:
    TraceFileParams traceParams_;
    std::unique_ptr<TraceFileReader> inputTraceFile_;
};

}}}

#endif
