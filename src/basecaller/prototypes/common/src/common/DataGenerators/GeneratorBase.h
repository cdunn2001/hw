#ifndef GENERATOR_BASE_H
#define GENERATOR_BASE_H

#include <dataTypes/TraceBatch.h>

#include <pacbio/PBException.h>

#include <vector_types.h>

#include <cstddef>
#include <vector>

namespace PacBio {
namespace Cuda {
namespace Data {

// Base class for generating data for use in a ZmwDataManager.  Child classes must
// implement the `PopulateBlock` function, which will be used to generate a block of data for
// caching.  The data generation is generally not necessary to be performant, though
// if it is a slow process one should manually call GenerateData() before starting
// the IO thread.  If it is not called manually then it will automatically be called
// upon the first invokation of `Fill`
template <typename T>
class GeneratorBase
{
public:
    GeneratorBase(size_t blockLen, size_t laneWidth, size_t numBlocks, size_t numZmwLanes)
        : blockLen_(blockLen)
        , laneWidth_(laneWidth)
        , numBlocks_(numBlocks)
        , numZmwLanes_(numZmwLanes)
    {}

    void GenerateData()
    {
        if (generated_) return;
        generatedData_.resize(numZmwLanes_);
        for (size_t i = 0; i < numZmwLanes_; ++i)
        {
            generatedData_[i].resize(numBlocks_);
            for (size_t j = 0; j < numBlocks_; j++)
            {
                generatedData_[i][j].resize(blockLen_ * laneWidth_);
                PopulateBlock(i, j, generatedData_[i][j]);
            }
        }
        generated_ = true;
    }

    void Fill(size_t laneIdx,
              size_t blockIdx,
              Mongo::Data::BlockView<T> v)
    {
        if (!generated_) GenerateData();

        if ((v.NumFrames() != blockLen_)  || (v.LaneWidth() != laneWidth_))
            throw PBException("Unexpectedly sized block received in GeneratorBase");

        const auto& data = generatedData_[laneIdx % numZmwLanes_][blockIdx % numBlocks_];
        assert(v.NumFrames()*v.LaneWidth() == data.size());
        std::copy(data.begin(), data.end(), v.Data());
    }

    virtual ~GeneratorBase() = default;

protected:
    size_t BlockLen() const { return blockLen_; }
    size_t LaneWidth() const { return laneWidth_; }
    size_t NumZmwLanes() const { return numZmwLanes_; }
    size_t NumBlocks() const { return numBlocks_; }

private:

    virtual void PopulateBlock(size_t laneIdx, size_t blockIdx, std::vector<T>& data) = 0;

    size_t blockLen_;
    size_t laneWidth_;
    size_t numBlocks_;
    size_t numZmwLanes_;
    bool generated_ = false;
    std::vector<std::vector<std::vector<T>>> generatedData_;
};

}}}

#endif
