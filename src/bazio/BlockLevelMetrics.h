// Copyright (c) 2018, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_POSTPRIMARY_BLOCKLEVELMETRICS_H
#define PACBIO_POSTPRIMARY_BLOCKLEVELMETRICS_H

#include <cassert>

#include "MetricData.h"
#include "RegionLabel.h"

#include <bazio/file/FileHeader.h>

namespace PacBio {
namespace Primary {

// Helper class, to access a portion of a vector without doing any copy/allocations
template <typename T>
class VectorView
{
public:
    // start is inclusive, end is exclusive
    VectorView(const std::vector<T>& data, size_t start, size_t end)
      : data_(data)
      , startIdx_(start)
      , endIdx_(end)
      {
          assert(start <= end);
          assert(end <= data.size());
      }

      size_t size() const { return endIdx_ - startIdx_; }
      bool empty() const { return endIdx_ == startIdx_; }
      const T& operator[](size_t idx) const
      {
          assert(idx < endIdx_);
          return data_.get()[startIdx_ + idx];
      }
      std::vector<T> copy() const { return std::vector<T>(data_.get().cbegin() + startIdx_, data_.get().cbegin() + endIdx_); }
      typename std::vector<T>::const_iterator cbegin() const { return data_.get().cbegin() + startIdx_; }
      typename std::vector<T>::const_iterator cend() const { return data_.get().cbegin() + endIdx_; }
private:
    std::reference_wrapper<const std::vector<T>> data_;
    size_t startIdx_;
    size_t endIdx_;
};

namespace detail {

// Helper functions, to create views of Analog/Filter data, which have internal
// vectors
template <typename T>
VectorView<T> CreateView(const std::vector<T>& data, size_t start, size_t end)
{
    return VectorView<T>(data, start, end);
}
template <typename T>
FilterMetricData<VectorView<T>> CreateView(const FilterMetricData<std::vector<T>>& data, size_t start, size_t end)
{
    // TODO risk accidental swap of arguments
    return FilterMetricData<VectorView<T>> {
    CreateView(data.green, start, end),
    CreateView(data.red, start, end) };
}
template <typename T>
AnalogMetricData<VectorView<T>> CreateView(const AnalogMetricData<std::vector<T>>& data, size_t start, size_t end)
{
    // TODO risk accidental swap of arguments
    return AnalogMetricData<VectorView<T>> {
    CreateView(data.A, start, end),
    CreateView(data.C, start, end),
    CreateView(data.T, start, end),
    CreateView(data.G, start, end) };
}

enum class MetricType
{
    SINGLE,  // Stands alone
    FILTER,  // has red/green components in sequel (spider still pending...)
    ANALOG   // has A,T,C,G components
};

template <typename T, MetricType Class> struct class_traits;
template <typename T> struct class_traits<T, MetricType::SINGLE> { using type = T; };
template <typename T> struct class_traits<T, MetricType::FILTER> { using type = FilterMetricData<T>; };
template <typename T> struct class_traits<T, MetricType::ANALOG> { using type = AnalogMetricData<T>; };

} // namespace detail

// Wrapping in a class now to slightly prepare for future changes where we
// have most metrics at LF, but a few at MF.  External code is going to be
// agnostic to all of those details
class MetricRegion
{
    size_t startBlock_;
    size_t endBlock_;

public:
    MetricRegion(size_t startBlock, size_t endBlock)
      : startBlock_(startBlock)
      , endBlock_(endBlock)
    {}

    size_t Start() const
    {
        return startBlock_;
    }

    size_t End() const
    {
        return endBlock_;
    }
};

// Stores a time series of metrics, as could be stored in the baz file.
template <typename T, detail::MetricType C>
class Metric
{
public:
    using Data_T = typename detail::class_traits<std::vector<T>, C>::type;
    using View_T = typename detail::class_traits<VectorView<T>, C>::type;

    Metric(const Metric&) = delete;
    Metric(Metric&&) = default;

    Metric& operator=(const Metric&) = delete;
    Metric& operator=(Metric&&) = default;

    Metric() = default;

    Metric(Data_T&& data, float frameRate, size_t framesPerBlock)
         : data_(std::move(data))
         , frameRate_(frameRate)
         , framesPerBlock_(framesPerBlock)
    {}

    float BlockTime() const
    {
        return framesPerBlock_ / frameRate_;
    }

    const View_T GetRegion(const MetricRegion& region) const { return GetRegion(region.Start(), region.End()); }
    const View_T GetRegionBefore(const MetricRegion& region) const { return GetRegion(0, region.Start()); }
    const View_T GetRegionAfter(const MetricRegion& region) const { return GetRegion(region.End(), data_.size()); }
    const View_T GetRegion(size_t start, size_t end) const { return detail::CreateView(data_, start, end); }

    const Data_T& data() const { return data_; }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

private:
    Data_T data_;
    float frameRate_;
    size_t framesPerBlock_;
};

template <typename T>
using SingleMetric = Metric<T, detail::MetricType::SINGLE>;
template <typename T>
using FilterMetric  = Metric<T, detail::MetricType::FILTER>;
template <typename T>
using AnalogMetric  = Metric<T, detail::MetricType::ANALOG>;

class RawMetricData;

// Class to contain all the metrics time series that come from the baz file.
class BlockLevelMetrics
{
public:
    // Needed for test code, as the Read object contains a BlockLevelMetrics
    // and itself is default constructed.  Remove once we can stop manually
    // altering what should be private fields.
    BlockLevelMetrics() = default;

    // The raw metrics are expected to come directly from the 
    // ParseMetrics(const FileHeader& fh, const ZmwByteData& data) function,
    // though they are accepted as an (unscaled) vector of vectors to remain
    // compatible with testing code that simulates these metrics.  rawMetrics is
    // expected to have one entry for each of MetricFieldNames, and each vector
    // it contains is expected to either be length 0 or the length consistent
    // with the number of metric blocks in the baz file.  The set of Metrics that
    // are populated needs to be consistent with one of the valid baz layouts,
    //
    // TODO: Once metrics computation has been refactored, remove `internal`
    //       flag from this class.
    BlockLevelMetrics(const RawMetricData& rawMetrics,
                      uint32_t metricFrames,
                      double frameRateHz,
                      const std::vector<float> relAmps,
                      const std::string& baseMap,
                      bool internal);

    /* const accessors */
    const SingleMetric<uint32_t>& NumFrames()             const { return numFrames_; }
    const SingleMetric<uint32_t>& NumPulsesAll()          const { return numPulsesAll_; }
    // Although pulse and base width are read from the BAZ file in
    // terms of frames, the same vectors are re-used to compute mean pulse and
    // and base width in terms of seconds so they are kept as float arrays.
    const SingleMetric<float>& PulseWidth()               const { return pulseWidth_; }
    const SingleMetric<float>& BaseWidth()                const { return baseWidth_; }
    const SingleMetric<uint32_t>& NumSandwiches()         const { return numSandwiches_; }
    const SingleMetric<uint32_t>& NumHalfSandwiches()     const { return numHalfSandwiches_; }
    const SingleMetric<uint32_t>& NumPulseLabelStutters() const { return numPulseLabelStutters_; }
    const SingleMetric<float>& PulseDetectionScore()      const { return pulseDetectionScore_; }
    const SingleMetric<float>& TraceAutocorr()            const { return traceAutocorr_; }

    const SingleMetric<uint32_t>& ActivityLabels()        const { return activityLabels_; }
    const SingleMetric<int32_t>& PixelChecksum()          const { return pixelChecksum_; }
    const SingleMetric<uint32_t>& DmeStats()              const { return dmeStats_; }

    const FilterMetric<float>& BaselineSD()     const { return baselineSD_; }
    const FilterMetric<float>& ChannelMinSNR()  const { return channelMinSNR_; }
    const FilterMetric<float>& BaselineMean()   const { return baselineMean_; }
    const FilterMetric<float>& Angle()          const {return angle_; }

    const AnalogMetric<float>& PkMax()       const { return pkMax_; }
    const AnalogMetric<float>& PkMid()       const { return pkMid_; }
    const AnalogMetric<uint32_t>& PkMidFrames() const {return pkMidFrames_; }
    const AnalogMetric<uint32_t>& NumBases()    const { return numBases_; }

    const AnalogMetric<float>& pkzvar() const { return pkzvar_; }
    const AnalogMetric<float>& bpzvar() const { return bpzvar_; }

    // These are derived metrics, not actually present in the baz file.
    // Debatable if they should be here or a separate class (though obviously
    // I leaned towards keeping them together)
    const SingleMetric<uint32_t>& NumBasesAll()     const { return numBasesAll_; }
    const SingleMetric<float>& BlockMinSNR()        const { return blockMinSNR_; }
    const SingleMetric<float>& BlockLowSNR()        const { return blockLowSNR_; }
    const SingleMetric<float>& MaxPkMaxNorms()      const { return maxPkMaxNorms_; }
    const SingleMetric<float>& PkzVarNorms()        const { return pkzVarNorms_; }
    const SingleMetric<float>& BpzVarNorms()        const { return bpzVarNorms_; }
    const AnalogMetric<float>& BasesVsTime()        const { return basesVsTime_; }

    // Converts the HQRegion (stored in terms of base/pulse index) into a
    // MetricRegion which defines a range within a Metric object
    MetricRegion GetMetricRegion(const RegionLabel& region) const;
    MetricRegion GetFullRegion() const;

    float StartTime(const RegionLabel& region) const
    {
        auto blockIdxs = GetMetricRegion(region);
        return blockIdxs.Start() * numPulsesAll_.BlockTime();
    }
    float StopTime(const RegionLabel& region) const
    {
        auto blockIdxs = GetMetricRegion(region);
        auto idx = blockIdxs.End();
        return idx * numPulsesAll_.BlockTime();
    }
private:
    SingleMetric<uint32_t> numFrames_;
    SingleMetric<uint32_t> numPulsesAll_;
    SingleMetric<float> pulseWidth_;
    SingleMetric<float> baseWidth_;
    SingleMetric<uint32_t> numSandwiches_;
    SingleMetric<uint32_t> numHalfSandwiches_;
    SingleMetric<uint32_t> numPulseLabelStutters_;
    SingleMetric<float> pulseDetectionScore_;
    SingleMetric<float> traceAutocorr_;

    FilterMetric<float> baselineSD_;
    FilterMetric<float> baselineMean_;
    FilterMetric<float> angle_;
    FilterMetric<float> channelMinSNR_;

    AnalogMetric<float> pkMax_;
    AnalogMetric<float> pkMid_;
    AnalogMetric<uint32_t> pkMidFrames_;
    AnalogMetric<uint32_t> numBases_;

    AnalogMetric<float> pkzvar_;
    AnalogMetric<float> bpzvar_;

    // optionally present in baz file
    SingleMetric<uint32_t> activityLabels_;

    // present in baz file, but unused in baz2bam...
    SingleMetric<int32_t> pixelChecksum_;
    SingleMetric<uint32_t> dmeStats_;

    // Derived, move elsewhere
    SingleMetric<uint32_t> numBasesAll_;
    SingleMetric<float> blockMinSNR_;
    SingleMetric<float> blockLowSNR_;
    SingleMetric<float> maxPkMaxNorms_;
    SingleMetric<float> pkzVarNorms_;
    SingleMetric<float> bpzVarNorms_;
    AnalogMetric<float> basesVsTime_;

    // Must store, so that we know if we should use pulses or bases when mapping
    // from HQRegion to a set of metric blocks
    bool internal_ = false;

    // Hack for now, to both avoid keeping a whole huge vector mapping pulses to
    // metric blocks, as well as to avoid recomputing this multiple times
    // despite the fact that everyone computes it for the same interval. Hopefully
    // remove this once metrics are refactored (and all HQ region stuff happens
    // at the same time)
    mutable std::pair<std::pair<int, int>, MetricRegion> cachedMap_ =
            std::make_pair(std::make_pair(0, 0), MetricRegion(0, 0));
    mutable bool mapSet_ = false;
};

}}

#endif /* BLOCKLEVELMETRICS_H */

