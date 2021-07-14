#ifndef mongo_basecaller_traceAnalysis_BlockFilterStage_H_
#define mongo_basecaller_traceAnalysis_BlockFilterStage_H_

#include "WindowBuffer.h"

#include <vector>

#include <common/LaneArray.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A block filter stage
// T: data type
// Filter: Filter class
template <typename T, typename Filter>
class BlockFilterStage
{
public: // Structors
    BlockFilterStage(size_t w, size_t s = 1, size_t nf = 1)
        : width_(w)
        , stride_(s)
        , nf_(nf)
        , winbuf_(w)
    {
        for (unsigned int i = 0; i < width_; ++i)
        {
            winbuf_.PushBack(T{0});
        }
    }

    BlockFilterStage(const BlockFilterStage& x) = delete;
    BlockFilterStage(BlockFilterStage&& x) = default;
    ~BlockFilterStage() = default;

    /// Transform input block to output block
    Data::BlockView<T>* operator()(Data::BlockView<T>* input);

private:
    const size_t width_;
    const size_t stride_;
    const size_t nf_;

    // Buffers
    WindowBuffer<LaneArray<T>> winbuf_;

    // Filter functor
    Filter filter_;
};

}}}

#endif // mongo_basecaller_traceAnalysis_BlockFilterStage_H_
