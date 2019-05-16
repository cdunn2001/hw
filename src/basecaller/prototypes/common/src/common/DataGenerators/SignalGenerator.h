#ifndef SIGNAL_GENERATOR_H
#define SIGNAL_GENERATOR_H

#include <common/ZmwDataManager.h>
#include <common/DataGenerators/GeneratorBase.h>
#include <common/DataGenerators/TraceFileReader.h>

#include <vector_types.h>

namespace PacBio {
namespace Cuda {
namespace Data {

using namespace PacBio::Mongo::Data;

struct TraceFileParams
{
    std::string traceFileName;

    TraceFileParams& TraceFileName(const std::string& v)    { traceFileName = v; return *this; }
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
        , traceFileReader_(std::make_unique<TraceFileReader>(traceParams_.traceFileName, params.zmwLaneWidth, params.blockLength))
    {}

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) override
    {
        traceFileReader_->PopulateBlock(laneIdx, blockIdx, v);
    }

private:
    TraceFileParams traceParams_;
    std::unique_ptr<TraceFileReader> traceFileReader_;
};

}}}

#endif
