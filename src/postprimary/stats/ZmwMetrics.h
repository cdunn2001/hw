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

#ifndef PACBIO_POSTPRIMARY_BAZSTS_ZMWMETRICS_H
#define PACBIO_POSTPRIMARY_BAZSTS_ZMWMETRICS_H

#include <bazio/MetricData.h>

#include <postprimary/insertfinder/InsertState.h>

// Not ideal includes, but they contain definitions for some
// pre-computed stuff we're going to slurp into our metrics.
#include <postprimary/bam/SubreadLabelerMetrics.h>
#include <postprimary/bam/EventData.h>

#include "ProductivityMetrics.h"

#include "PBOptional.h"


namespace PacBio {
namespace Primary {
namespace Postprimary {

class SignalMetrics
{
    template <typename T>
    using FilterM = FilterMetricData<T>;
    template <typename T>
    using AnalogM = AnalogMetricData<T>;

public:
    SignalMetrics() = default;
    SignalMetrics(const MetricRegion& region,
                  const BlockLevelMetrics& metrics);
    
    const FilterM<float>& Baseline() const { return baseline_; }
    const FilterM<float>& BaselineSD() const { return baselineSD_; }
    const FilterM<float>& MinSnr() const { return minSnr_; }
    const AnalogM<float>& PkMid() const { return pkMid_; }
    const AnalogM<float>& Snr() const { return snr_; }
    const AnalogM<float>& Angle() const { return angle_; }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(SignalMetrics);
    }

private:
    FilterM<float> baseline_ = {0, 0};
    FilterM<float> baselineSD_ = {0, 0};
    FilterM<float> minSnr_ = {0, 0};
    AnalogM<float> pkMid_ = {0, 0, 0, 0};
    AnalogM<float> snr_ = {0, 0, 0, 0};
    AnalogM<float> angle_ = {0, 0, 0, 0};
};

class BaseMetrics
{
public:
    BaseMetrics() = default;
    BaseMetrics(const FileHeader& fh,
                const RegionLabel& hqRegion,
                const BlockLevelMetrics& metrics,
                const EventData& events);
    
    float Width() const { return width_; }
    float Rate() const { return rate_; }
    float LocalRate() const { return localRate_; }
    float Ipd() const { return ipd_; }
    float HQRatio() const { return HQRatio_; }
    float Pausiness() const { return pausiness_; }

    const AnalogMetricData<uint32_t>& Counts() const { return counts_; }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(BaseMetrics);
    }

private:
    float width_ = 0;
    float rate_ = 0;
    float localRate_ = 0;
    float ipd_ = 0;
    float HQRatio_ = 0;
    float pausiness_ = 0;
    AnalogMetricData<uint32_t> counts_ = {0, 0, 0, 0};
};

class PulseMetrics
{
public:
    PulseMetrics() = default;
    PulseMetrics(const FileHeader& fh,
                 const RegionLabel& hqRegion,
                 const BlockLevelMetrics& metrics);

    float Width() const { return width_; }
    float Rate() const { return rate_; }
    uint32_t TotalCount() const { return totalCount_; }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(PulseMetrics);
    }

private:
    float width_ = 0;
    float rate_ = 0;
    uint32_t totalCount_ = 0;
};

class ExcludedPulseMetrics
{
public:
    ExcludedPulseMetrics() = default;
    ExcludedPulseMetrics(const std::vector<InsertState>& insertStates);

    struct InsertData {
        uint32_t base = 0;
        uint32_t exShortPulse = 0;
        uint32_t burstPulse = 0;
        uint32_t pausePulse = 0;
    };

    const InsertData& InsertCounts() const { return insertCounts_; }
    const InsertData& InsertLengths() const { return insertLengths_; }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(ExcludedPulseMetrics);
    }

private:
    InsertData insertCounts_;
    InsertData insertLengths_;
};

class ReadMetrics
{
public:
    ReadMetrics() = default;
    ReadMetrics(const FileHeader& fh,
                const RegionLabel& hqRegion,
                const EventData& events,
                const ProductivityInfo& prod);
    
    uint32_t UnitFeatures() const { return unitFeatures_; }
    uint32_t ReadLength() const { return readLength_; }
    uint32_t PolyLength() const { return polyLength_; }
    uint32_t HoleNumber() const { return holeNumber_; }
    bool Internal() const { return internal_; }
    float IsRead() const { return isRead_; }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(ReadMetrics);
    }

private:
    uint32_t unitFeatures_ = 0;
    uint32_t readLength_ = 0;
    uint32_t polyLength_ = 0;
    uint32_t holeNumber_ = 0;
    bool internal_ = false;
    float isRead_ = false;
};

class SubreadMetrics
{
public:
    SubreadMetrics() = default;
    SubreadMetrics(const RegionLabel& hqRegion,
                   std::vector<RegionLabel> adapterRegions);

    double MeanLength() const { return meanLength_; }
    double MaxSubreadLength() const { return maxSubreadLength_; }
    double MedianLength() const { return medianLength_; }
    double ReadScore() const { return readScore_; }
    double Umy() const { return umy_; }

    size_t EstimatedBytesUsed() const
    {
        return sizeof(SubreadMetrics);
    }

private:
    double meanLength_ = 0;
    double maxSubreadLength_ = 0;
    double medianLength_ = 0;
    double readScore_ = 0;
    double umy_ = 0;
};

class ZmwMetrics
{
private:
    // defined for all zmw
    SignalMetrics zmwSignalMetrics_;
    ExcludedPulseMetrics excludedPulseMetrics_;
    ReadMetrics readMetrics_;

    // Only exists if HQ region exists
    PBOptional<BaseMetrics> baseMetrics_;
    PBOptional<PulseMetrics> pulseMetrics_;
    PBOptional<SignalMetrics> hqSignalMetrics_;
    PBOptional<SubreadMetrics> subreadMetrics_;

    // Compted elsewhere but carried along for convenience
    ProductivityInfo prodMetrics_;
    ControlMetrics controlMetrics_;
    AdapterMetrics adapterMetrics_;

public:
    ZmwMetrics(const FileHeader& fh,
               const RegionLabel& hqRegion,
               const std::vector<RegionLabel>& adapters,
               const BlockLevelMetrics& metrics,
               const EventData& events,
               const ProductivityInfo& prod,
               const ControlMetrics& control,
               const AdapterMetrics& adapterMetrics);

    ZmwMetrics(const RegionLabel& hqRegion,
               const std::vector<RegionLabel>& adapters,
               const ProductivityInfo& prod,
               const ControlMetrics& control,
               const AdapterMetrics& adapterMetrics,
               bool computeInsertStats);

    const SignalMetrics& ZmwSignalMetrics() const { return zmwSignalMetrics_; }
    const ExcludedPulseMetrics& ZmwExcludedPulseMetrics() const { return excludedPulseMetrics_; }
    const ReadMetrics& ZmwReadMetrics() const { return readMetrics_; }
    const PBOptional<PulseMetrics>& ZmwPulseMetrics() const { return pulseMetrics_; }
    const PBOptional<BaseMetrics>& ZmwBaseMetrics() const { return baseMetrics_; }
    const PBOptional<SignalMetrics>& ZmwHqSignalMetrics() const { return hqSignalMetrics_; }
    const PBOptional<SubreadMetrics>& ZmwSubreadMetrics() const { return subreadMetrics_; }
    const ProductivityInfo& ZmwProdMetrics() const { return prodMetrics_; }
    const ControlMetrics& ZmwControlMetrics() const { return controlMetrics_; }
    const AdapterMetrics& ZmwAdapterMetrics() const { return adapterMetrics_; }

    size_t EstimatedBytesUsed() const
    {
        // The PBOptional objects store an actual value so we just use sizeof() here.
        // All of the AnalogMetricData and FilterMetricData use either floating or
        // integral data types so using sizeof() avoids double counting that
        // would occur if we use EstimatedBytesUsed().
        return sizeof(ZmwMetrics);
    }
};

}}} //PacBio::Primary::Postprimary

#endif /* PACBIO_POSTPRIMARY_BAZSTS_ZMWMETRICS_H */

