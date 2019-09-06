// Copyright (c) 2017, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// THIS SOFTWARE CONSTITUTES AND EMBODIES PACIFIC BIOSCIENCES' CONFIDENTIAL
// AND PROPRIETARY INFORMATION.
//
// Disclosure, redistribution and use of this software is subject to the
// terms and conditions of the applicable written agreement(s) between you
// and Pacific Biosciences, where "you" refers to you or your company or
// organization, as applicable.  Any other disclosure, redistribution or
// use is prohibited.
//
// THIS SOFTWARE IS PROVIDED BY PACIFIC BIOSCIENCES AND ITS CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//  Description:
//  Defines a compile time sparse matrix.  You can specify via template
//  parameters which row/col indexes are present.  It's main benefit is that
//  it will provide compile time information about the data layout, allowing
//  efficient traversal patterns that the compiler can potentially optimize
//  aggresively

#ifndef Sequel_Basecaller_Common_SparseMatrix2_H_
#define Sequel_Basecaller_Common_SparseMatrix2_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <type_traits>
#include <vector>

namespace PacBio {
namespace Cuda {

// Simple container types to keep track of a list of template parameters all of
// the same type.
template <bool...> struct BoolList {};
template <size_t... vals> struct IndexList {};

// Simple type to encode a compact segment in a sparse row.  The first
// parameter indicates how many nonzero entries precede this segment in the
// row (meaning it can be used as the index of the first column into a compactly
// stored version of the row), where the next two indicate the begin and end
// (exclusive) column indicies for this run of nonzero entries.  It is valid
// to encode a run of 1 (firstCol_ = lastCol_ - 1)
template <size_t dataIndex_, size_t firstCol_, size_t lastCol_>
struct CompactRowSegment
{
    static_assert(lastCol_ > firstCol_, "Invalid CompactRowSegment specification");

    static constexpr size_t dataIndex = dataIndex_;
    static constexpr size_t firstCol = firstCol_;
    static constexpr size_t lastCol = lastCol_;
};

// Helper function to inspect that a template parameter is a given type.
// T is any parameter you want to examine and U is the expected template template
// parameter (i.e. BoolList, IndexList or CompactRowSegment).  Manually specifying
// the value type I is unfortunate, but I couldn't find a clean way to make
// the compiler deduce it.
template <class T, typename I, template<I...> class U>
class IsTypeOf
{
    // Fallback case if T takes the wrong template arguments, or isn't even
    // a template template parameter to begin with.
    template <typename TT>
    struct helper { static constexpr bool value = false; };
    // This only matches if the compiler can successfully find a template
    // parameter pack in T and plug them into U, meaning T is of the
    // expected type.
    template <I... vals>
    struct helper<U<vals...>> { static constexpr bool value = true; };
public:
    static constexpr bool value = helper<T>::value;
};

// TMP function type to see if a list of bools are all true.  Used for validation
// checks in various static_assert statements
template <class T> class AllTrue {
    // Make sure input parameter is expected type
    static_assert(IsTypeOf<T, bool, BoolList>::value, "AllTrue requires a valid BoolList as input");

    // Extract the bools as a template pack and inspect them.
    template <class U> struct helper;
    template <bool... bs>
    struct helper<BoolList<bs...>> {
        // If this was called with a bool list that only contains true values,
        // then the list will be the same if we append a true to either end.  If
        // the list contains any false values then the prepended type will have to
        // be different from the appended type.
        static constexpr bool value = std::is_same<BoolList<true, bs...>, BoolList<bs..., true>>::value;
    };
public:
    static constexpr bool value = helper<T>::value;
};

// TMP function that appends a size_t to an IndexList
template <size_t val, typename List>
class AppendIndex
{
    // Make sure List really is an IndexList
    static_assert(IsTypeOf<List, size_t, IndexList>::value, "AppendIndex can only operate in IndexLists");

    // Expect the values as a parameter pack, and combine with val.
    template <class U> struct helper;
    template <size_t... oldVals>
    struct helper<IndexList<oldVals...>>
    {
        using type = IndexList<oldVals..., val>;
    };

public:
    using type = typename helper<List>::type;
};

// TMP fuction to generate an IndexList that goes from 0 to len-1
template <size_t len>
class MakeIndexList
{
    // Recursive helper to loop over each required value and append to the list
    template <size_t next, size_t end, class CurrList>
    struct helper
    {
        using type = typename helper<next+1, end, typename AppendIndex<next, CurrList>::type>::type;
    };
    // End of recursion.
    template <size_t end, class FinishedList>
    struct helper<end, end, FinishedList>
    {
        using type = FinishedList;
    };
public:
    using type = typename helper<0, len, IndexList<>>::type;
};

// TMP function that accepts a list of CompactRowSegments, and sums their lengths
template <class... Segments>
class SumSegmentWidths
{
    // Validate each of our input arguments
    using valid_args = BoolList<IsTypeOf<Segments, size_t, CompactRowSegment>::value...>;
    static_assert(AllTrue<valid_args>::value, "Not all arguments to SumSegmenntWidths are CompactRowSegments");

    // Recursive helper to loop over the argument lists.
    template <size_t sum, class... Args>
    struct sum_helper;
    template <size_t sum, class Head, class... Tail>
    struct sum_helper<sum, Head, Tail...> {
        static constexpr size_t value = sum_helper<sum + Head::lastCol - Head::firstCol, Tail...>::value;
    };
    template <size_t sum>
    struct sum_helper<sum> {
        static constexpr size_t value = sum;
    };
public:
    static constexpr size_t value = sum_helper<0, Segments...>::value;
};

// Simple struct to serve a runtime version of CompactRowSegment.
struct CompactSegment
{
    constexpr CompactSegment(size_t pDataIndex, size_t pFirstCol, size_t pLastCol)
        : dataIndex (pDataIndex)
        , firstCol (pFirstCol)
        , lastCol (pLastCol)
    { }

    const size_t dataIndex; /* Number of nonzero entries in the row before firstCol*/
    const size_t firstCol;  /* Start column for contiguous batch of entries */
    const size_t lastCol;   /* First zero entry after current run */
};

// This is our main representation for a compile time sparse row.  It is
// specified by a row ID, the first nonzero index, and then a list of
// contiguous segments of length 1 or greater.
//
// While it is possible to manually specify a SparseRow type and it will validate
// its own inputs, it is recommended you use the `Row` class below.
template <size_t row, size_t firstIndex, class... Segments>
class SparseRow
{
    // Make sure our variadic pack has the expected types.
    using valid_args = BoolList<IsTypeOf<Segments, size_t, CompactRowSegment>::value...>;
    static_assert(AllTrue<valid_args>::value, "Invalid template arguments to SparseRow");

    // We need to make sure that our list of Segments describes a valid row.
    // Here we check that our dataIndex parameters are consistent, as it is
    // meant to record the number of nonzero entries before the current segment.
    //
    // To use this helper, we have to slightly massage the Segments data.
    // If SegmentWidths is the width of Segment i, Idxs1 will be the dataIndex
    // of Segment i-1, and Idxs2 will be the same for Segment i.  In doing this,
    // the dataIndex values are valid if and only if the difference between
    // idx2 and idx1 is the same as segmentwidths.
    template <typename idxs1, typename idxs2, typename SegmentWidths>
    struct validate_segments;
    template <size_t... idxs1, size_t... idxs2, size_t... SegmentWidths>
    struct validate_segments<IndexList<idxs1...>, IndexList<idxs2...>, IndexList<SegmentWidths...>>
    {
        using valid = BoolList<(idxs2 - idxs1 == SegmentWidths)...>;
        static constexpr bool value = AllTrue<valid>::value;
    };

    // Need to compute the sum of all the widths, as we'll have to insert this
    // at the end of Idxs2
    static constexpr size_t segment_sum = SumSegmentWidths<Segments...>::value;

    // We're going to expand our Segments into 3 IndexLists.  The first two
    // are the start and end index of the segment into a dense storage
    // representation for the row, and the last is the width of each segment.
    // Note that for the purposes of this check, `firstIndex` is added as a
    // segment of width 1 at the beginning.
    static_assert(validate_segments<
            IndexList<0, Segments::dataIndex...>,
            IndexList<Segments::dataIndex..., segment_sum+1>,
            IndexList<1, (Segments::lastCol - Segments::firstCol)...>
            >::value, "Inconsistent column segments in SparseRow");

    // Helper class to loop over each Segment, and make sure that they are
    // ordered correctly.
    template <size_t maxCol, class...args>
    struct validate;
    template <size_t maxCol, class idx1, class idx2, class... tail>
    struct validate<maxCol, idx1, idx2, tail...>
    {
        // Note: The definition of SparseRowSegment itself ensures that
        //       lastCol > firstCol
        static constexpr bool increasing =
                idx1::firstCol > firstIndex // Ensure that firstIndex really is first
             && idx1::dataIndex < idx2::dataIndex // Ensure they are ordered in terms of storage
             && idx1::lastCol < idx2::firstCol  // Ensure they are increasing in terms of column indexes
             && validate<maxCol, idx2, tail...>::increasing; // Recursion

        // Make sure no one is out of bounds for the entire row.
        static constexpr bool inBounds = (idx1::lastCol < maxCol)
                                      && validate<maxCol, idx2, tail...>::inBounds;
    };
    // End of recursion
    template <size_t maxCol, class idx>
    struct validate<maxCol, idx>
    {
        static constexpr bool increasing = true;
        static constexpr bool inBounds = idx::lastCol <= maxCol;
    };

public:
    // Have to wrap these checks in constexpr function because they depend
    // on maxCol (to be provided by the matrix type) and c++11 doesn't
    // have template variables yet.  Will always return true, the actual work
    // is done by the static asserts.
    template <size_t maxCol>
    static constexpr bool ValidRow()
    {
        using validator = validate<maxCol, Segments...>;
        static_assert(validator::increasing, "Sparse indexes must be monotonically increasing");
        static_assert(validator::inBounds, "Sparse index is too large");
        return true;
    }

    static constexpr size_t firstIdx = firstIndex;
    static constexpr size_t rowIdx = row;
    static constexpr size_t nElements = segment_sum+1;

    // Convert compile time CompactRowSegments into runtime CompactSegments
    // so they can be stored in the same std::array
    static constexpr std::array<CompactSegment, sizeof...(Segments)> segments =
        {{CompactSegment(Segments::dataIndex, Segments::firstCol, Segments::lastCol)...}};

    // Helper function to convert array to vector.  Useful when manipulating
    // multiple rows which likely have different length arrays.
    static std::vector<CompactSegment> SegmentVector()
    {
        return std::vector<CompactSegment>(segments.begin(), segments.end());
    }
};

// Class to represent sparse matrix, with the sparsity pattern encoded via
// template parameters.  It does provide row/col access via the operator(),
// but the sparsity is enforced.  You can read from any element, but the class
// will trigger an assert if you attempt to write to an entry that has been
// marked as zero.
//
// The operator() function is relatively inefficient, not suitable for performance
// critical sections.  Normal accesses should occur via the RowData function,
// using the classes template information for efficient access processing only
// the nonzero entries.
//
// Note: This class currently assumes values are stored in logspace.  They don't
//       have to be in logspace when initially populating it, but if so, the
//       log() function should be called before accessing as investigating
//       the 0 elements will return -std::numeric_limits<float>::infinity()
//
// Note: As with SparseRow, you can manually specify the template parameters,
//       but it is recommended that you instead use the Matrix class below as
//       it has a more intuitive interface.
template <class... Rows>
class SparseMatrix
{
public:

private:
    // Make sure that each row supplied has a valid specification
    using valid_rows = BoolList<Rows::template ValidRow<sizeof...(Rows)>()...>;
    static_assert(AllTrue<valid_rows>::value, "Invalid rows present");


public:
    SparseMatrix() = default;

    std::array<std::vector<size_t>, sizeof...(Rows)> InitMatrix()
    {
        // Keeps track of which columns are nonzero in each row
        std::array<std::vector<size_t>, sizeof...(Rows)> idxLists;

        // Matrix is assumed square.  Validation static_asserts will trigger if
        // supplied template arguments are inconsistent with this.
        static constexpr size_t nRows = sizeof...(Rows);

        // Keep track of how many nonzero entries each row has.
        static constexpr std::array<size_t, sizeof...(Rows)> rowLengths_ = {{Rows::nElements...}};

        // Compute the start index for each row into our compact storage array.
        rowStartIdx_[0] = 0;
        for (size_t i = 1; i < nRows; i++)
        {
            rowStartIdx_[i] = rowStartIdx_[i-1] + rowLengths_[i-1];
        }

        // Allocate enough space to hold our nonzero entries
        std::fill(data_, data_+13*13, 0.0f);

        // Helper function, that takes a row specification (runtime version)
        // and expands it to a list of present columns.
        auto fill = [](size_t first, std::vector<CompactSegment> columnSegments)
        {
            std::vector<size_t> ret;
            ret.push_back(first);
            for (const auto& segment : columnSegments)
            {
                for (auto start = segment.firstCol; start < segment.lastCol; start++)
                {
                    ret.push_back(start);
                }
            }
            return ret;
        };

        // Fill out the idxLists for each row in the matrix
        auto implicit_loop = {(idxLists[Rows::rowIdx] = fill(Rows::firstIdx, Rows::SegmentVector()),0)...};
        (void)implicit_loop;

        return idxLists;
    }

    // Read/write method for examining an entry.  Will throw an assert if you
    // access one of the zero entries!
    // Note this is a relatively inefficient method, as it has to search for
    // the mapping between row/col and sparse format storage.
    half& operator()(size_t row, size_t col,
                      const std::array<std::vector<size_t>, sizeof...(Rows)>& idxLists)
    {
        assert(row < sizeof...(Rows));
        assert(col < sizeof...(Rows));

        auto colPtr = std::find(idxLists[row].begin(), idxLists[row].end(), col);
        assert(colPtr != idxLists[row].end());
        return data_[rowStartIdx_[row] + std::distance(idxLists[row].begin(), colPtr)];
    }

    half& Entry(size_t row, size_t col,
                const std::array<std::vector<size_t>, sizeof...(Rows)>& idxLists)
    {
        return (*this)(row, col, idxLists);
    }

    // Read only access to entries.  Can access any row/col in the matrix, but
    // it will return -inf (not 0) for the sparse entries.
    // Note this is a relatively inefficient method, as it has to search for
    // the mapping between row/col and sparse format storage.
    half operator()(size_t row, size_t col,
                     const std::array<std::vector<size_t>, sizeof...(Rows)>& idxLists) const
    {
    // Matrix is assumed square.  Validation static_asserts will trigger if
    // supplied template arguments are inconsistent with this.
    static constexpr size_t nRows = sizeof...(Rows);
    static constexpr size_t nCols = sizeof...(Rows);
        assert(row < nRows);
        assert(col < nCols);

        auto colPtr = std::find(idxLists[row].begin(), idxLists[row].end(), col);
        if (colPtr == idxLists[row].end())
            return -std::numeric_limits<float>::infinity();
        else
            return data_[rowStartIdx_[row] + std::distance(idxLists[row].begin(), colPtr)];
    }

    //size_t size() const
    //{
    //    return data_.size();
    //}

    // Grants read-only access to the raw row data.  It is the most efficient
    // access as there is no extra layer of indirection, but requires you use
    // the information encoded in the template parameters to find specific
    // entries.
    //
    // As the template parameters encode the first index, and then a sequence
    // of contiguous columns, it is assumed you will use data[0] to initialize
    // data, and then loop over each segment individually to move through the
    // raw data.
    __host__ __device__ const half* RowData(size_t row) const
    {
        assert(row < sizeof...(Rows));
        return &data_[rowStartIdx_[row]];
    }

    // This matrix is designed to operate in log space, but will allow you to
    // input values in normal space and convert the whole matrix afterwards.
    // void log()
    // {
    // // Matrix is assumed square.  Validation static_asserts will trigger if
    // // supplied template arguments are inconsistent with this.
    // static constexpr size_t nRows = sizeof...(Rows);
    // static constexpr size_t nCols = sizeof...(Rows);
    //     // First validate data to make sure we can take the log.
    //     for (size_t i = 0; i < data_.size(); ++i)
    //     {
    //         // Not technically illegal, but surprising given the current usage.
    //         // Feel free to remove if you intentionally set a value to zero
    //         // without marking it as such in the template specifications
    //         assert(data_[i] != 0.0f);

    //         if (data_[i] < 0.0f)
    //         {
    //             std::stringstream msg;
    //             msg << "Found negative entries in sparse matrix, cannot convert to log space!:\n";
    //             PrintMatrix(msg);
    //             throw PBException(msg.str());
    //         }
    //     }

    //     for (size_t i = 0; i < data_.size(); ++i)
    //     {
    //         using std::log;
    //         data_[i] = log(data_[i]);
    //     }
    // }

private:
    // For printing the matrix during a validation error.  Assumes data has
    // not been converted to logspace (e.g. the sparse values that normally are
    // returned as -inf are printed as 0.0f)
    void PrintMatrix(std::ostream& msg) const
    {
    // Matrix is assumed square.  Validation static_asserts will trigger if
    // supplied template arguments are inconsistent with this.
    static constexpr size_t nRows = sizeof...(Rows);
    static constexpr size_t nCols = sizeof...(Rows);
        // Set up pretty formatting
        auto currFlags = msg.flags();
        msg << std::scientific << std::setprecision(3);

        // Need const ref to access version that does not assert when accessing
        // sparse entries
        const auto& self = *this;
        for (size_t i = 0; i < nRows; i++)
        {
            for (size_t j = 0; j < nCols; j++)
            {
                auto val = self(i,j);
                if (val == -std::numeric_limits<float>::infinity())
                    val = 0.0f;
                // The width flag isn't sticky so we have to set it for each
                // entry...
                msg << std::setw(10) << val << " " ;
            }
            msg << "\n";
        }

        // Restore original formating.
        msg.flags(currFlags);
    }
    // Keeps track of where each row starts in the data_ array
    size_t rowStartIdx_[sizeof...(Rows)];

    // Raw compact storage for the whole matrix.
    half data_[sizeof...(Rows)*sizeof...(Rows)];

};

// Alternate (and preferred) specification for a Row.  You just manually type
// out a sequence of 1 and 0, specifying which columns are present.  A bit more
// verbose than how you specify a SparseRow, but a lot more intuitive and readable.
template <size_t... colFlags>
class SparseRowSpec
{
public:
    static constexpr size_t nCols = sizeof...(colFlags);
private:
    // Make sure only 1 and 0 are input.  Could have templated the class on a
    // list of bools, but writing out true/false is too verbose, and would allow
    // something like `6` to be recorded as `true`, which might mislead readers
    // into thinking that a `6` vs a `1` is meaningful
    using valid_cols = BoolList<(colFlags == 0 || colFlags == 1)...>;
    static_assert(AllTrue<valid_cols>::value, "Can only use 0 or 1 in sparsity matrix!");

    using entries = BoolList<colFlags...>;

    // Helper class to take a sequence of bools and find either the next true
    // or next false (specified by `match`).  The return value will be idx +
    // the distance into `cols` that the match was made. If no match was made then
    // `found` will be `false` and `idx` will be one past the end of the list.
    template <size_t idx, bool match, class List>
    struct find_next_bool;

    template <size_t idx, bool match, bool Head, bool... Tail>
    struct find_next_bool<idx, match, BoolList<Head, Tail...>>
    {
        // Declare our recursive type, even if we actually want to stop here
        using next_t = find_next_bool<idx+1, match, BoolList<Tail...>>;
        // Check if we found the type we're looking for, and either return or continue
        static constexpr bool found = (match == Head) ? true : next_t::found;
        // Same, but record the index we found
        static constexpr size_t value = (match == Head) ? idx : next_t::value;
        // Record the list of bools that remains after we finish.  This way,
        // if we called this ones starting from the begining of a list of n bools
        // (i.e. idx = 0), then if this returns saying the first true is at
        // index 6, we return here the last n-6 bools, allowing us to start
        // looping again from index 7 to find the next false after the current
        // true.
        using tail = typename std::conditional<match == Head, BoolList<Tail...>, typename next_t::tail>::type;
    };
    // Need to cap off our recursion.
    template <size_t idx, bool match>
    struct find_next_bool<idx, match, BoolList<>>
    {
        static constexpr bool found = false;
        static constexpr size_t value = idx;
        using tail = BoolList<>;
    };

    // Helper class to loop over all the entries in the row, and gather them into
    // the firstIdx + CompactRowSegments required to specify a SparseRow.
    // Idx                    - our current index in our overall loop
    // currDataIdx            - the index into the raw row array for the segment we're curretly finding
    // DataIdx, Begin and End - IndexLists describing all the segments we've found so far
    // Entries                - Remaning bools to loop over
    template <size_t idx, size_t currDataIdx, class DataIdx, class Begin, class End, class Entries>
    struct AccumulateSegments
    {
        // Find the next true and subsequent false, to define our new segment range
        // If there is no next true, then NextStart::found will be false
        // If there is a next true, then NextEnd will point to the first false
        // after that, even if it is an implicit false one past the end of our array
        using NextStart = find_next_bool<idx, true, Entries>;
        using NextEnd = find_next_bool<NextStart::value+1, false, typename NextStart::tail>;

        // Compute our updated Lists.  If we found another segment then we'll
        // have to continue for another iteration.  If we did not find another
        // segment then expose our Begin etc template parameters instead.
        using updatedBegin   = typename std::conditional<NextStart::found, typename AppendIndex<NextStart::value, Begin>::type, Begin>::type;
        using updatedEnd     = typename std::conditional<NextStart::found, typename AppendIndex<NextEnd::value, End>::type  , End>::type;
        using updatedCompact = typename std::conditional<NextStart::found, typename AppendIndex<currDataIdx, DataIdx>::type   , DataIdx>::type;

        // Compute our next recursive type
        using NextRecurse = AccumulateSegments<
                NextEnd::value,     //Index starts 1 after last found segment (can be 1 past end of array)
                currDataIdx + NextEnd::value - NextStart::value,  //Increment our index into the raw data array
                updatedCompact, updatedBegin, updatedEnd,         // Pass on updated lists
                typename NextEnd::tail>;                          // Only iterate over unprocessed bools

        // Either continue the recursion, or expose our final return types
        using BeginPack   = typename std::conditional<NextStart::found, typename NextRecurse::BeginPack,   Begin>::type;
        using EndPack     = typename std::conditional<NextStart::found, typename NextRecurse::EndPack,     End>::type;
        using CompactPack = typename std::conditional<NextStart::found, typename NextRecurse::CompactPack, DataIdx>::type;
    };
    // Terminate our recursion.  Even if the loop above terminated, these
    // types will still be invoked and we need a specialization to terminate.
    template <size_t idx, size_t currCompact, class Compact, class Begin, class End>
    struct AccumulateSegments<idx, currCompact, Compact, Begin, End, BoolList<>>
    {
        using BeginPack = Begin;
        using EndPack = End;
        using CompactPack = Compact;
    };

    // Make sure we have at least one entry
    using first_t = find_next_bool<0, true, entries>;
    static_assert(first_t::found, "Found sparse row with no entries");

public:

    static constexpr size_t firstIdx = first_t::value;
    using Accumulation = AccumulateSegments<firstIdx+1, 1, IndexList<>, IndexList<>, IndexList<>, typename first_t::tail>;
};

// Helper class to transform a Row class into a SparseRow class.  I would
// have preferred it to be an inner class hidden in the Row class itself, but
// intel had problems with that for some reason (gcc was fine)
template <size_t rowIdx, class Row>
struct SparseRowGenerator
{

    template <class Compact, class Begin, class End>
    struct sparse_helper;
    template <size_t... Compact, size_t... Begin, size_t...End>
    struct sparse_helper<IndexList<Compact...>, IndexList<Begin...>, IndexList<End...>>
    {
        using type = SparseRow<rowIdx, Row::firstIdx, CompactRowSegment<Compact, Begin, End>...>;
    };

    using Accumulation = typename Row::Accumulation;
    using type = typename sparse_helper<
            typename Accumulation::CompactPack,
            typename Accumulation::BeginPack,
            typename Accumulation::EndPack>::type;
};

// Helper class used to transform a list of Rows into an actual SparseMatrix
// type
template <class... Rows>
class SparseMatrixGenerator
{
    static constexpr size_t nRows = sizeof...(Rows);
    static constexpr size_t nCols = sizeof...(Rows);

    // Make sure our matrix is square
    using valid_rows = BoolList<(Rows::nCols == nCols)...>;
    static_assert(AllTrue<valid_rows>::value, "Specified matrix is not square or has irregular row lengths!");

    template <typename IdxList, typename...Rs>
    struct Create;
    template <size_t... idxs, typename... Rs>
    struct Create<IndexList<idxs...>, Rs...>
    {
        static_assert(sizeof...(idxs) == sizeof...(Rs), "wtd");
        using type = SparseMatrix<typename SparseRowGenerator<idxs, Rs>::type...>;
    };

public:
    using type = typename Create<typename MakeIndexList<sizeof...(Rows)>::type, Rows...>::type;
};

// This is the recommended interface for declaring a sparse matrix.
// Usage will look something like:
// SparseMatrixSpec<
//   SparseRowSpec<1,1,1,1>,
//   SparseRowSpec<1,0,1,1>,
//   SparseRowSpec<0,0,1,0>
//   SparseRowSpec<1,0,0,0>>;
//
// It will fail with validation asserts if the matrix is not square or if
// any rows are entirely sparse.  Otherwise, it will automatically convert
// all the lists of 1/0 into the structures expected by the SparseMatrix
// type.
template <class... Rows>
using SparseMatrixSpec = typename SparseMatrixGenerator<Rows...>::type;

// definitions for static member declarations
template <size_t row, size_t firstIdx, class...Segments>
constexpr std::array<CompactSegment,sizeof...(Segments)> SparseRow<row, firstIdx, Segments...>::segments;

    using Transition_t = SparseMatrixSpec<
    //                    B  T  G  C  A  TU GU CU AU TD GD CD AD
            SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // Baseline
            SparseRowSpec<0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>, // T
            SparseRowSpec<0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>, // G
            SparseRowSpec<0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0>, // C
            SparseRowSpec<0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0>, // A
            SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // T Up
            SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // G Up
            SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // C Up
            SparseRowSpec<1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1>, // A Up
            SparseRowSpec<0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0>, // T Down
            SparseRowSpec<0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0>, // G Down
            SparseRowSpec<0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0>, // C Down
            SparseRowSpec<0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0>  // A Down
        >;

}}

#endif /* Sequel_Basecaller_Common_SparseMatrix_H_ */
