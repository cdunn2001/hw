#ifndef SAWTOOTH_GENERATOR_H
#define SAWTOOTH_GENERATOR_H

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/ZmwDataManager.h>
#include <common/DataGenerators/GeneratorBase.h>

#include <vector_types.h>

#include <algorithm>
#include <cassert>
#include <cstddef>

namespace PacBio {
namespace Cuda {
namespace Data {

class SawtoothGenerator : public GeneratorBase<short2>
{
public:
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    SawtoothGenerator(const DataManagerParams& params)
        : GeneratorBase(params.blockLength, params.gpuLaneWidth, params.numBlocks, 1)
        , gpuLaneWidth_{params.gpuLaneWidth}
        , blockLen_{params.blockLength}
    {}

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<short2>& v) override
    {
        assert(gpuLaneWidth_ * blockLen_ == v.size());
        for (size_t i = 0; i < blockLen_; ++i)
        {
            short tmp = (i + blockIdx * blockLen_) % SawtoothHeight;
            short2 val = {tmp, tmp};
            auto ptr1 = v.data() + i * gpuLaneWidth_;
            auto ptr2 = ptr1 + gpuLaneWidth_;
            std::fill(ptr1, ptr2, val);
        }
    }
public:

    void ValidateData(const Memory::HostView<short2>& view, const DataManagerParams& params,
                      size_t blockIdx, size_t batchIdx)
    {
        bool valid = true;
        short frameOffset = blockIdx * params.blockLength;
        for (size_t i = 0; i < params.kernelLanes; ++i)
        {
            size_t blockOffset = i * params.blockLength * params.gpuLaneWidth;
            for (size_t j = 0; j < params.blockLength; ++j)
            {
                if (frameOffset+j < FilterWidth) continue;

                size_t rowOffset = blockOffset + j*params.gpuLaneWidth;
                short expectedVal = (frameOffset + j - 1) % SawtoothHeight;
                if (expectedVal < FilterWidth-1) expectedVal = SawtoothHeight-1;
                {
                    valid &= (view[rowOffset].x == expectedVal);
                }
            }
        }
        if (!valid)
            std::cerr << "batch " << batchIdx << " failed validation\n";
    }

private:
    size_t gpuLaneWidth_;
    size_t blockLen_;
};


}}}

#endif
