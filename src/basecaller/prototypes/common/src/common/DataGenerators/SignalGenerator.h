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

class SignalGenerator : public GeneratorBase<int16_t>
{
public:
    SignalGenerator(const DataManagerParams& params, const TraceFileParams& traceParams)
        : GeneratorBase(params.blockLength,
                        params.laneWidth,
                        params.numBlocks,
                        params.numZmwLanes)
        , traceParams_(traceParams)
        , traceFileReader_(traceParams_.traceFileName, params.laneWidth, params.blockLength)
    {}

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<int16_t>& v) override
    {
        traceFileReader_.PopulateBlock(laneIdx, blockIdx, v);
    }

private:
    TraceFileParams traceParams_;
    TraceFileReader traceFileReader_;
};

}}}

#endif
