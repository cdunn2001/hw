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

class SawtoothGenerator : public GeneratorBase<int16_t>
{
public:
    static constexpr int FilterWidth = 7;
    static constexpr int SawtoothHeight = 25;

    SawtoothGenerator(const DataManagerParams& params)
        : GeneratorBase(params.blockLength, params.laneWidth, params.numBlocks, 1)
        , blockLen_{params.blockLength}
    {}

private:
    void PopulateBlock(size_t laneIdx,
                       size_t blockIdx,
                       std::vector<int16_t>& v) override
    {
        assert(LaneWidth() * blockLen_ == v.size());
        for (size_t i = 0; i < blockLen_; ++i)
        {
            int16_t val = (i + blockIdx * blockLen_) % SawtoothHeight;
            auto ptr1 = v.data() + i * LaneWidth();
            auto ptr2 = ptr1 + LaneWidth();
            std::fill(ptr1, ptr2, val);
        }
    }
public:

    void ValidateData(const Memory::HostView<int16_t>& view, const DataManagerParams& params,
                      size_t blockIdx, size_t batchIdx)
    {
        bool valid = true;
        int16_t frameOffset = blockIdx * params.blockLength;
        for (size_t i = 0; i < params.kernelLanes; ++i)
        {
            size_t blockOffset = i * params.blockLength * params.laneWidth;
            for (size_t j = 0; j < params.blockLength; ++j)
            {
                if (frameOffset+j < FilterWidth) continue;

                size_t rowOffset = blockOffset + j*params.laneWidth;
                int16_t expectedVal = (frameOffset + j - 1) % SawtoothHeight;
                if (expectedVal < FilterWidth-1) expectedVal = SawtoothHeight-1;
                {
                    valid &= (view[rowOffset] == expectedVal);
                }
            }
        }
        if (!valid)
            std::cerr << "batch " << batchIdx << " failed validation\n";
    }

private:
    size_t blockLen_;
};


}}}

#endif
