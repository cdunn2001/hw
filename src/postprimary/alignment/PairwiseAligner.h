// Copyright (c) 2014-2016, Pacific Biosciences of California, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the
// disclaimer below) provided that the following conditions are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//  * Neither the name of Pacific Biosciences nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE
// GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY PACIFIC
// BIOSCIENCES AND ITS CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL PACIFIC BIOSCIENCES OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
// USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

#pragma once

#include <algorithm>
#include <assert.h>
#include <stdlib.h>
#include <utility>
#include <cmath>
#include <vector>
#include <iomanip>
#include <cassert>

#include <bazio/SequenceUtilities.h>

#include "Cell.h"
#include "ScoringScheme.h"

namespace PacBio {
namespace Primary {
namespace Postprimary {

/// Various pairwise aligner and auxiliary methods.
class PairwiseAligner
{
public:  // Structors
    // Default constructor
    PairwiseAligner() = default;
    // Move constructor
    PairwiseAligner(PairwiseAligner&&) = delete;
    // Copy constructor is deleted!
    PairwiseAligner(const PairwiseAligner&) = delete;
    // Move assignment operator
    PairwiseAligner& operator=(PairwiseAligner&&) = default;
    // Copy assignment operator is deleted!
    PairwiseAligner& operator=(const PairwiseAligner&) = delete;
    // Destructor
    ~PairwiseAligner() = default;

public:  // Smith-Waterman alignment functions

    /// Traverse the last row of an SW matrix (i.e. representing
    ///     alignments terminating with the last base of the query
    ///     sequence) and return the max score and it's position
    ///
    /// \param  matrix       pointer to SW matrix
    /// \param  queryLength  length of the query sequence
    /// \param  readLength   length of the read sequence
    ///
    /// \return  A std::pair of the max score and it's position
    static std::pair<int32_t, int32_t> SWLastRowMax(
            const int32_t* matrix,
            const int32_t queryLength,
            const int32_t readLength)
    {
        // Calculate the starting position of the last row
        const int32_t M = queryLength + 1;
        const int32_t N = readLength + 1;
        const int32_t beginLastRow = (M - 1) * N;

        // Find maximal score in last row and it's position
        int32_t maxScore = -1;
        int32_t endPos = 0;
        for (int32_t j = 0; j < N; ++j)
        {
            if (matrix[beginLastRow + j] > maxScore)
            {
                maxScore = matrix[beginLastRow + j];
                endPos = j;
            }
        }

        // Return the maximum score and position as a pair
        return std::make_pair(maxScore, endPos);
    }

    /// Backtracking of a SW matrix from a given cell that contains the
    /// column cell.jEndPosition of interest and the last row.
    /// \param matrix      pointer to SW matrix
    /// \param cell        Cell contains the starting column and serves as
    ///                    an information holder for results
    /// \param N           Width of the matrix, needed to address indices
    /// \param query       query string to compute (mis)matches
    /// \param read        read string to compute (mis)matches
    /// \param scoring     ScoringScheme for DP algorithm
    static void SWBacktracking(const int32_t* matrix, Cell& cell,
                               const int32_t N, const std::string& query,
                               const std::string& read,
                               const ScoringScheme& scoring,
                               const int32_t stopRow = 0)
    {
        return SWBacktracking(matrix, cell, N, query.size(), query.c_str(),
                              read.c_str(), scoring, true, stopRow);
    }

    /// Computes and provides pointer to a filled SW matrix.
    /// \param  query       query string
    /// \param  read        read string
    /// \param  scoring     ScoringScheme for DP algorithm
    /// \param  resetAtRow  Reset negative scores to 0 at specified row (useful
    ///                     to allow two "global" alginments in one matrix)
    /// \return             int32_t* to the SW matrix
    static int32_t* SWComputeMatrix(const std::string& query,
                                    const std::string& read,
                                    const ScoringScheme& scoring,
                                    const int resetAtRow = 0) noexcept
    {
        return SWComputeMatrix(query.c_str(), query.size(),
                               read.c_str(),  read.size(),
                               scoring,       true, resetAtRow);
    }

    /// \brief Computes and provides pointer to a filled SW matrix.
    /// \param  query         query string
    /// \param  read          read string
    /// \param  scoring       ScoringScheme for DP algorithm
    /// \param  globalInQuery bool for whether to penalize partial query hits
    /// \return               int32_t* to the SW matrix
    static int32_t* SWComputeMatrix(const std::string& query,
                                    const std::string& read,
                                    const ScoringScheme& scoring,
                                    const bool globalInQuery) noexcept
    {
        return SWComputeMatrix(query.c_str(), query.size(),
                               read.c_str(),   read.size(),
                               scoring,     globalInQuery);
    }

    /// Computes and provides pointer to a filled SW matrix.
    /// \param  query       char* to the query
    /// \param  queryLength Length of the query array
    /// \param  read        char* to the read
    /// \param  readLength  Length of the read array
    /// \param  scoring     ScoringScheme for DP algorithm
    /// \return             int32_t* to the SW matrix
    static int32_t* SWComputeMatrix(const char* const query,
                                    const int32_t queryLength,
                                    const char* const read,
                                    const int32_t readLength,
                                    const ScoringScheme& scoring,
                                    const bool globalInQuery,
                                    const int resetAtRow = 0) noexcept
    {
        const int32_t M = queryLength + 1;
        const int32_t N = readLength + 1;

        int32_t* matrix = new int32_t[M * N];

        SWComputeMatrix(query, M, read, N, scoring, globalInQuery, matrix, resetAtRow);

        return matrix;
    }

    /// Computes and provides pointer to a filled SW matrix.
    /// \param  query       char* to the query
    /// \param  queryLength Length of the query array
    /// \param  read        char* to the read
    /// \param  readLength  Length of the read array
    /// \param  scoring     ScoringScheme for DP algorithm
    /// \return             int32_t* to the SW matrix
    static int32_t* SWComputeMatrix(const char* const query,
                                    const int32_t queryLength,
                                    const char* const read,
                                    const int32_t readLength,
                                    const ScoringScheme& scoring) noexcept
    {
        const int32_t M = queryLength + 1;
        const int32_t N = readLength + 1;

        int32_t* matrix = new int32_t[M * N];

        SWComputeMatrix(query, M, read, N, scoring, true, matrix);

        return matrix;
    }

    /// Fills out a supplied SW matrix.
    /// \param  query       char* to the query
    /// \param  queryLength Length of the query array
    /// \param  read        char* to the read
    /// \param  readLength  Length of the read array
    /// \param  scoring     ScoringScheme for DP algorithm
    /// \param  matrix      int32_t* to the SW matrix
    static inline void SWComputeMatrix(const char* const query,
                                const int32_t M,
                                const char* const read,
                                const int32_t N,
                                const ScoringScheme& scoring,
                                const bool globalInQuery,
                                int32_t*& matrix,
                                const int resetAtRow = 0) noexcept
    {
        matrix[0] = 0;

        if (globalInQuery)
            for (int32_t i = 1; i < M; ++i)
                matrix[i * N] = i * scoring.deletionPenalty;
        else
            for (int32_t i = 1; i < M; ++i)
                matrix[i * N] = 0;

        for (int32_t j = 1; j < N; ++j)
            matrix[j] = 0;

        char iQuery;
        char iBeforeQuery;
        int32_t mismatchDelta = scoring.matchScore - scoring.mismatchPenalty;
        int32_t insertionDelta = scoring.branchPenalty - scoring.insertionPenalty;
        for (int32_t i = 1; __builtin_expect(i < M, 1); ++i)
        {
            iQuery = query[i];
            iBeforeQuery = query[i - 1];
            if (__builtin_expect(i < M - 1, 1))
            {
                for (int32_t j = 1; __builtin_expect(j < N, 1); ++j)
                {
                    // branch = match && read[j - 2] == read[j - 1];
                    int32_t a = matrix[(i - 1) * N + j - 1] + scoring.matchScore;
                    int32_t b = matrix[i * N + j - 1] + scoring.branchPenalty;
                    int32_t c = matrix[(i - 1) * N + j] + scoring.deletionPenalty;
                    if (read[j - 1] != iBeforeQuery)
                        a -= mismatchDelta;
                    if (read[j - 1] != iQuery)
                        b -= insertionDelta;
                    if (!globalInQuery || i == resetAtRow)
                        matrix[i * N + j] = std::max({0, a, b, c});
                    else
                        matrix[i * N + j] = std::max({a, b, c});
                }
            }
            else
            {
                for (int32_t j = 1; __builtin_expect(j < N, 1); ++j)
                {
                    // branch = match && read[j - 2] == read[j - 1];
                    int32_t a = matrix[(i - 1) * N + j - 1] + scoring.matchScore;
                    int32_t b = matrix[i * N + j - 1] + scoring.insertionPenalty;
                    int32_t c = matrix[(i - 1) * N + j] + scoring.deletionPenalty;
                    if (read[j - 1] != iBeforeQuery)
                        a -= mismatchDelta;
                    if (!globalInQuery || i == resetAtRow)
                        matrix[i * N + j] = std::max({0, a, b, c});
                    else
                        matrix[i * N + j] = std::max({a, b, c});
                }
            }
        }
    }

    /// Backtracking of a SW matrix from a given cell that contains the
    /// column cell.jEndPosition of interest.
    /// \param matrix           pointer to SW matrix
    /// \param cell             Cell contains the starting column and serves as
    ///                         an information holder for results
    /// \param N                Width of the matrix, needed to address indices
    /// \param row              Starting row of backtracking
    /// \param query            char* to the query to compute (mis)matches
    /// \param read             char* to the read to compute (mis)matches
    /// \param scoring          ScoringScheme for DP algorithm
    /// \param globalInQuery    Backtrack to full extent of query
    /// \param stopRow          Backtrack to exactly stopRow (if globalInQuery)
    static void SWBacktracking(const int32_t* matrix, Cell& cell,
                               const int32_t N, const int32_t row,
                               const char* const query, const char* const read,
                               const ScoringScheme& scoring,
                               const bool globalInQuery,
                               const int32_t stopRow = 0)
    {
        const int32_t minValue = std::numeric_limits<int32_t>::min();

        int32_t i = row;
        int32_t j = cell.jEndPosition;
        while (i >= (stopRow + 1) && j >= 1)
        {
            // What is the current Cell's score, and where might it be from?
            int32_t score = matrix[i * N + j];
            if (score <= 0 && !globalInQuery)
                break;
            int32_t matchValue  = matrix[(i - 1) * N + j - 1];
            int32_t insertValue = matrix[i * N + j - 1];
            int32_t deleteValue = matrix[(i - 1) * N + j];

            // What are the differences between the score and it's parents?
            int32_t matchDiff  = score - matchValue;
            int32_t insertDiff = score - insertValue;
            int32_t deleteDiff = score - deleteValue;

            // Could this position be a Match? A Branch?
            bool isMatch = query[i - 1] == read[j - 1];
            bool isBranch = read[j - 1] == query[i];
            int32_t expDiagDiff = isMatch ? scoring.matchScore : scoring.mismatchPenalty;
            int32_t expVerticalPenalty = isBranch ? scoring.branchPenalty : scoring.insertionPenalty;

            // If the score difference for a move isn't possible, set the score
            //    to an arbitrary low value so we don't use it.
            matchValue  = (matchDiff  == expDiagDiff)             ? matchValue  : minValue;
            insertValue = (insertDiff == expVerticalPenalty)      ? insertValue : minValue;
            deleteValue = (deleteDiff == scoring.deletionPenalty) ? deleteValue : minValue;

            // This is a little complicated. If we're running a matrix that has
            // been modified to be semiglobal at certain rows, we'll run into
            // moves that cannot be explained, as we're moving essentially
            // starting scoring over. Lets reconstitute the match and delete
            // values and skip the insert option (as it will be 0
            if (!(max3(matchValue, insertValue, deleteValue) > minValue))
            {
                matchValue  = matrix[(i - 1) * N + j - 1];
                deleteValue = matrix[(i - 1) * N + j];
                insertValue = minValue;
            }

            // Switch on the values from above
            switch (argmax3(matchValue, insertValue, deleteValue))
            {
            case 0:
                if (query[i - 1] == read[j - 1])
                    ++cell.matches;
                else
                    ++cell.mismatches;
                --i;
                --j;
                break;
            case 1:
                ++cell.insertions;
                --j;
                break;
            case 2:
                ++cell.deletions;
                --i;
                break;
            }
        }
        cell.iBeginPosition = i;
        cell.jBeginPosition = j;
    }

    /// Compute the maximal Smith-Waterman score
    static int32_t SWComputeMaxScore(const std::string& query,
                                     const std::string& read,
                                     const ScoringScheme& scoring) noexcept
    {
        return SWComputeMaxScore(query.c_str(), query.size(),
                                 read.c_str(), read.size(), scoring);
    }

private:
    /// Compute the maximal Smith-Waterman score
    static int32_t SWComputeMaxScore(const char* const query, const int32_t queryLength,
                                     const char* const read, const int32_t readLength,
                                     const ScoringScheme& scoring) noexcept
    {
        const int32_t M = queryLength;
        const int32_t N = readLength;

        const int32_t colNum = M + 1;
        const int32_t rowSize = N + 1;

        int32_t* matrix = SWComputeMatrix(query, M, read, N, scoring);

        int32_t max = -1;
        for (int32_t k = 0; k < colNum*rowSize; ++k)
            if (matrix[k] > max) max = matrix[k];

        delete[] matrix;

        return max;
    }


public:  // Flanking Sequence Functions
    /// Computes the identity score of the flanking regions of an adapter.
    /// \param  read     read string
    /// \param  cell     Cell reference to get offsets
    /// \param  scoring  ScoringScheme for DP algorithm
    static void FlankingScore(const std::string& read, Cell& cell,
                              const ScoringScheme& scoring,
                              bool localAln, uint32_t flankLength) noexcept
    {
        return FlankingScore(read.c_str(), read.size(), cell, scoring,
                             localAln, flankLength);
    }

    /// Computes the identity score of the flanking regions of an adapter.
    /// \param  read       char* to the read
    /// \param  readLength Length of the query array
    /// \param  cell       Cell reference to get offsets
    /// \param  scoring    ScoringScheme for DP algorithm
    static void FlankingScore(const char* const read, const int32_t readLength,
                              Cell& cell, const ScoringScheme& scoring,
                              bool localAln, uint32_t flankLength) noexcept
    {
        flankLength = std::min(cell.jBeginPosition, flankLength);
        flankLength = std::min(readLength - cell.jEndPosition, flankLength);

        // Too small, let's trust it blindly
        if (flankLength < 10)
        {
            cell.flankingScore = 1000;
        }
        else // Otherwise align up to flankLength bases from either end of the adapter to each other
        {
            char* rc = new char[flankLength + 1];
            PacBio::Primary::SequenceUtilities::ReverseCompl(&read[cell.jEndPosition], flankLength, rc);
            if (localAln)
            {
                cell.flankingScore = PairwiseAligner::SWComputeMaxScore(
                    &read[cell.jBeginPosition - flankLength + 1], flankLength,
                    rc, flankLength, scoring);
            }
            else
            {
                cell.flankingScore = PairwiseAligner::NWComputeMaxScore(
                    &read[cell.jBeginPosition - flankLength + 1], flankLength,
                    rc, flankLength, scoring);
            }
            delete[] rc;
        }
    }

public:  // Matrix Display Functions
    /// Prints a supplied S-W matrix to stdout
    /// \param  matrix      int32_t* to the SW matrix
    /// \param  query       char* to the query
    /// \param  queryLength Length of the query array
    /// \param  read        char* to the read
    /// \param  readLength  Length of the read array
    /// \return void
    static void PrintMatrix(const int32_t* const matrix,
                            const char* const query,
                            const int32_t M,
                            const char* const read,
                            const int32_t N)
    {
        int32_t colNum = M + 1;
        int32_t rowSize = N + 1;

        // Print out the header, consisting of the bases in Read
        std::cout << "      0";
        for (int32_t j = 1; __builtin_expect(j < rowSize, 1); ++j)
        {
            std::cout << std::setw(4) << read[j - 1];
        }
        std::cout << std::endl;

        for (int32_t i = 0; i < colNum; ++i)
        {
            // Print out a Null for the first row
            if (i == 0)
            {
                std::cout << " 0 ";
            } 
            // ..And the appropriate Query base for each successive row
            else
            {
                std::cout << " " << query[i - 1] << " ";
            }

            // Print out each column J in row I of the matrix
            for (int32_t j = 0; j < rowSize; ++j)
            {
                std::cout << std::setw(4) << matrix[i * rowSize + j];
            }
            // End the Row
            std::cout << std::endl;
        }
        // End the Matrix
        std::cout << std::endl;

        return;
    }

    /// Prints a supplied S-W matrix to stdout
    /// \param  matrix      int32_t* to the SW matrix
    /// \param  query       std::string of the query
    /// \param  read        std::string of the read
    static void PrintMatrix(const std::string queryStr,
                            const std::string readStr,
                            const int32_t* const matrix)
    {
        // Convert our std::strings to char arrays
        const char* query = queryStr.c_str();
        const int32_t M   = queryStr.length();
        const char* read  = readStr.c_str();
        const int32_t N   = readStr.length();

        // Call the main PrintMatrix func and return
        PrintMatrix(matrix, query, M, read, N);
        return;
    }

// Needleman-Wunsch Alignment functions
public:
    /// Computes the maximal Needleman-Wunsch score
    /// from the last row and column of the DP matrix.
    /// \param query    query string&
    /// \param read     read string&
    /// \param scoring  coringScheme for DP algorithm
    /// \return Maximal NW score, last row and last column.
    static int32_t NWComputeMaxScore(const std::string& query,
                                     const std::string& read,
                                     const ScoringScheme& scoring) noexcept
    {
        return NWComputeMaxScore(query.c_str(), query.size(), read.c_str(),
                                 read.size(), scoring);
    }

private:

    /// Computes the maximal Needleman-Wunsch score
    /// from the last row of the DP matrix.
    /// \param query    query string&
    /// \param read     read string&
    /// \param scoring  coringScheme for DP algorithm
    /// \return Maximal NW score, last row.
    static int32_t NWComputeMaxLastRowScore(const std::string& query,
                                            const std::string& read,
                                            const ScoringScheme& scoring) noexcept
    {
        return NWComputeMaxLastRowScore(query.c_str(), query.size(),
                                        read.c_str(), read.size(), scoring);
    }


    static int32_t NWComputeMaxScore(const char* const query, const int32_t queryLength,
                                     const char* const read, const int32_t readLength,
                                     const ScoringScheme& scoring) noexcept
    {
        const int32_t M = queryLength;
        const int32_t N = readLength;

        const int32_t colNum = M + 1;
        const int32_t rowSize = N + 1;

        int32_t* matrix = NWComputeMatrix(query, M, read, N, scoring);

        int32_t max = -1;
        for (int32_t i = 0; i < colNum; ++i)
            if (matrix[i * rowSize + N] > max) max = matrix[i * rowSize + N];
        for (int32_t j = 0; j < rowSize; ++j)
            if (matrix[j + M * rowSize] > max) max = matrix[j + M * rowSize];

        delete[] matrix;

        return max;
    }

    static int32_t NWComputeMaxLastRowScore(const char* const query, const int32_t queryLength,
                                            const char* const read, const int32_t readLength,
                                            const ScoringScheme& scoring) noexcept
    {
        const int32_t M = queryLength + 1;
        const int32_t N = readLength + 1;

        int32_t* matrix = NWComputeMatrix(query, M, read, N, scoring);

        int32_t max = -1;
        for (int j = 0; j + (M - 1) * N < M * N; ++j)
            if (matrix[j + (M - 1) * N] > max) max = matrix[j + (M - 1) * N];

        delete[] matrix;

        return max;
    }

    static int32_t* NWComputeMatrix(const char* const query, const int32_t M,
                                    const char* const read, const int32_t N,
                                    const ScoringScheme& scoring) noexcept
    {
        int32_t colNum  = M + 1;
        int32_t rowSize = N + 1;
        int32_t* matrix = new int32_t[colNum * rowSize];

        matrix[0] = 0;

        for (int32_t i = 1; __builtin_expect(i <= M, 1); ++i)
            matrix[i * rowSize] = scoring.deletionPenalty * i;

        for (int32_t j = 1; __builtin_expect(j <= N, 1); ++j)
            matrix[j] = scoring.insertionPenalty * j;

        for (int32_t i = 1; __builtin_expect(i <= M, 1); ++i)
        {
            for (int32_t j = 1; __builtin_expect(j <= N, 1); ++j)
            {
                matrix[(i * rowSize) + j] =
                    max3((matrix[((i - 1) * rowSize) + j - 1] +
                         ((query[i - 1] == read[j - 1]) ? scoring.matchScore : scoring.mismatchPenalty)),
                         matrix[(i * rowSize) + j - 1] + scoring.insertionPenalty,
                         matrix[((i - 1) * rowSize) + j] + scoring.deletionPenalty);
            }
        }

        return matrix;
    }

private:  // Utility functions
    static inline int max2(int a, int b) { return (a > b) ? a : b; }

    static inline int max3(int a, int b, int c)
    {
        return (a > b) ? ((a > c) ? a : c) : ((c > b) ? c : b);
    }

    static inline char max3path(int a, int b, int c)
    {
        return (a > b) ? ((a > c) ? 'A' : 'C') : ((c > b) ? 'C' : 'B');
    }

    static inline int argmax3(int a, int b, int c)
    {
        return (a > b) ? ((a > c) ? 0 : 2) : ((c > b) ? 2 : 1);
    }

};

}}} // ::PacBio::Primary::Postprimary
