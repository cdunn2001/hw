#ifndef TEMPLATE_GENERATOR_H
#define TEMPLATE_GENERATOR_H

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/DataGenerators/GeneratorBase.h>
#include <common/ZmwDataManager.h>

#include <algorithm>
#include <cstddef>

namespace PacBio {
namespace Cuda {
namespace Data {

// Just a dummy generator to demonstrate the API.
struct TemplateGenerator : public GeneratorBase<short>
{
    TemplateGenerator(const DataManagerParams& params)
        : GeneratorBase(params.blockLength, params.laneWidth, params.numBlocks, params.numZmwLanes)
    {}

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short>& v) override
    {
        // just cram a dummy value that will vary as we change lane and block
        short val = laneIdx * blockIdx;
        std::fill(v.data(), v.data() + v.size(), val);
    }
};

}}}

#endif
