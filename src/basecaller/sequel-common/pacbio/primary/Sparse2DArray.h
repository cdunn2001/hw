//
// Created by mlakata on 1/11/18.
//

#ifndef PPA_SPARSE2DARRAY_H
#define PPA_SPARSE2DARRAY_H

#include <stdint.h>
#include <map>
#include <vector>

namespace PacBio {
namespace Primary {


/// This defines a sparse 2D array which is mostly a single value, but has a small number of values that are different.
/// In addition, the values tend to be clustered, i.e. a lot of repeated values near the borders. For this purpose,
/// the implemention stores the array as a map (using the y index) that point to vectors of "Spans" of constant value,
/// of template type T.
/// The Span is defined as a start and stop index in the x dimension, and a constant value for that span.
/// The look up of a (x,y) value is done by looking up the Span vector for index y, then linearly searching the
/// vector for the x value. A linear search is used because in this application, the number of spans is small (usually
/// less than 10 per value of y).
/// If nothing is found, the default value of the array is returned.
template<typename T>
class Sparse2DArray
{
    struct Span
    {
    public:
        int16_t startIndex; // inclusive
        int16_t endIndex;   // inclusive
        T value;
    };
public:
    Sparse2DArray() = default;

    /// Construct sparse array from individual values.
    /// Each tuple of (xes[i],yes[i],values[i]) defines a value of a particular coordinate of the sparse
    /// array.
    /// \param numValues the size of the xes, yes, and values arrays
    /// \param xes The x coordinates of the values
    /// \param yes The y coordinates of the values
    /// \param values The values corresponding to the xes and yes of the same index.
    /// \param defaultValue the value of most of the sparse array that is not covered by the particular values.
    void CompressSparseArray(int numValues, const int16_t xes[], const int16_t yes[], const T values[],
                               const T defaultValue)
    {
        defaultValue_ = defaultValue;

        if (numValues > 0)
        {
            // this assumes that the iteration is Y-major (i.e. X is inner loop)
            std::vector<Span> row;
            Span thisSpan;

            thisSpan.startIndex = xes[0];
            thisSpan.value = values[0];
            int16_t lastX = xes[0];
            int16_t lastY = yes[0];

            //std::cout << "i:" << 0 << " x:" << xes[0] << " y:" << yes[0] << " value:" << (int)values[0] <<std::endl;

            for (int i = 1; i < numValues; i++)
            {
                const int16_t x = xes[i];
                const int16_t y = yes[i];
                const T value = values[i];

                //std::cout << "i:" << i << " x:" << x << " y:" << y << " value:" << (int)value <<std::endl;

                // row change or value change (or flushing because we're at the end)
                if (x != lastX+1 || y != lastY || value != thisSpan.value)
                {
                    // flush span
                    thisSpan.endIndex = lastX ;
                    row.push_back(thisSpan);

                    //std::cout << "y:" << y << " span:" << thisSpan.start_ << ":" << thisSpan.endIndex << " value:"
                    //        << (int) thisSpan.value << std::endl;

                    // row change
                    if (y != lastY)
                    {
                        // flush row
                        if (row.size() > 0)
                        {
                            //std::cout << "row: y:" << lastY << " count:" << row.size() << std::endl;
                            map_[lastY] = row;
                        }
                        row.clear();
                    }

                    // start a new span
                    thisSpan.startIndex = x;
                    thisSpan.value = value;
                }

                lastX = x;
                lastY = y;
            }

            thisSpan.endIndex = lastX ;
            row.push_back(thisSpan);

            //std::cout << "y:" << lastY << " span:" << thisSpan.startIndex << ":" << thisSpan.endIndex << " value:"
            //        << (int) thisSpan.value << std::endl;

            //std::cout << "row: y:" << lastY << " count:" << row.size() << std::endl;
            map_[lastY] = row;
        }
    }

    // look up value
    T Value(int16_t x, int16_t y) const
    {
        auto rowIter = map_.find(y);
        if (rowIter != map_.end())
        {
            const std::vector<Span>& row = rowIter->second;
            for (const auto& span : row)
            {
                if (x >= span.startIndex && x <= span.endIndex)
                {
                    return span.value;
                }
            }
        }
        return defaultValue_;
    }
    
#if 0
    /// this is an approximation of the size used by the sparse array, for benchmarking purposes.
    size_t TotalMemoryUsed() const
    {
        return sizeof(decltype(map_)::value_type)* map_.size() + sizeof(map_);
    }
#endif

private:
    T defaultValue_;
    std::map<int16_t /* x */, std::vector<Span>> map_;
};

}} // end namespaces

#endif //PPA_SPARSE2DARRAY_H
