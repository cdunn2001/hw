#ifndef mongo_basecaller_traceanalysis_WindowBuffer_H_
#define mongo_basecaller_traceanalysis_WindowBuffer_H_

#include <vector>
#include <tbb/concurrent_queue.h>
#include <boost/circular_buffer.hpp>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

template <typename T>
class WindowBuffer : public boost::circular_buffer<T, tbb::cache_aligned_allocator<T>>
{
public:
    WindowBuffer(size_t w = 0)
        : boost::circular_buffer<T, tbb::cache_aligned_allocator<T>>(w)
        , counter_(0)
        , fVal_(T{})
    { }

public:
    void PushBack(typename WindowBuffer<T>::param_value_type item = WindowBuffer<T>::value_type())
    {
        this->push_back(item);
        counter_ = (counter_ + 1) % this->capacity();
    }

    /// Set a filter-specific value held across chunks
    void SetHoldoverValue(T val) { fVal_ = val; }

public:
    int Counter() const { return counter_; }

    /// Get a filter-specific value held across chunks
    T GetHoldoverValue() const { return fVal_; }

private:
    int counter_;
    T fVal_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceanalysis_WindowBuffer_H_
