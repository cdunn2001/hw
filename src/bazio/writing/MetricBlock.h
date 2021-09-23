// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_METRICIO_WRITING_METRIC_BLOCK_H
#define PACBIO_METRICIO_WRITING_METRIC_BLOCK_H

#include <bazio/MetricBlock.h>

namespace PacBio::BazIO {

/// CRTP to represent a metric block used by the underlying
/// metric buffer manager and metric buffer classes. This
/// allows for the actual metrics to be de-coupled from the
/// bazio library.
///
template <typename T>
struct MetricBlock
{
    /// Performs aggregation of the input metric block.
    template <typename U>
    void Aggregate(const MetricBlock<U>& val)
    {
        static_assert(std::is_base_of<MetricBlock<T>,T>::value, "T must be derived from MetricBlock<T>!");
        static_cast<T*>(this)->Aggregate(static_cast<const U&>(val));
    }

    /// Returns the activity label associated with this metric block.
    /// Used to determine whether metric blocks can be aggregated.
    uint8_t ActivityLabel() const
    {
        return static_cast<const T*>(this)->ActivityLabel();
    }

    /// Returns true if metric block contains data.
    bool HasData() const
    {
        return static_cast<const T*>(this)->HasData();
    }

    /// Stub method for converting to the current output metric block
    /// format in use by the BAZ file. This should be removed once
    /// the BAZ metrics have been ported to use the bazio encoding
    /// framework.
    void Convert(Primary::SpiderMetricBlock& sm) const
    {
        static_cast<const T*>(this)->Convert(sm);
    }

};

} // PacBio::BazIO

#endif  // PACBIO_METRICIO_WRITING_METRIC_BLOCK_H


