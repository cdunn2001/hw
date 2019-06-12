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
template <typename V, typename F>
class BlockFilterStage
{
public: // Structors
    BlockFilterStage(size_t w, size_t s = 1)
        : width_(w)
        , leftHalfWidth_(w / 2)
        , rightHalfWidth_(1 + (w - 1) / 2)
        , stride_(s)
        , winbuf_(w)
        , rhsDeficit_(rightHalfWidth_)
    { }

    BlockFilterStage(const BlockFilterStage& x) = delete;
    BlockFilterStage(BlockFilterStage&& x) = default;
    ~BlockFilterStage() = default;

    /// Transform input block to output block
    Data::BlockView<V>* operator()(Data::BlockView<V>* input);

private:
    const size_t width_;
    const size_t leftHalfWidth_;
    const size_t rightHalfWidth_;
    const size_t stride_;

    // Filter functor
    F filter_;

    // Buffers
    WindowBuffer<LaneArray<V>> winbuf_;
    int rhsDeficit_;
};

}}}

#endif // mongo_basecaller_traceAnalysis_BlockFilterStage_H_
