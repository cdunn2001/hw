// Copyright (c) 2015, Pacific Biosciences of California, Inc.
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
/// \brief  buffer that contains the output from the t2b process for one ZMW

#pragma once

#include <cstdint>
#include <array>

#include <pacbio/logging/Logger.h>
#include <pacbio/smrtdata/Basecall.h>

#include "BasecallingMetrics.h"
#include "TrancheSizes.h"
#include "PrimaryConfig.h"

using PacBio::SmrtData::Basecall;

namespace PacBio {
namespace Primary {

/// A buffer to hold the results for a single ZMW.
/// 
template <typename TFeat>
class ZmwResultBuffer
{
public:

    // This class should be used to initialize a ZmwResultBuffer object so
    // that the memory could be correctly assigned. The ZmwResultBuffer's constructor
    // is deleted to prevent creating on the heap/stack.
    class Initialize
    {
    public:
        static ZmwResultBuffer<TFeat>* FromMemoryLocation(void* memPtr)
        {
            ZmwResultBuffer<TFeat>* r = static_cast<ZmwResultBuffer<TFeat>*>(memPtr);
            r->UpdatePointers();
            return r;
        }
    };

public: // Static constants
    static constexpr size_t MaxMetricsSize = std::max(sizeof(BasecallingMetricsT::Sequel),
                                                      sizeof(BasecallingMetricsT::Spider));

    static constexpr size_t MaxMetricsLatency = 2;

public: // Static members and methods
    static void Configure(const PrimaryConfig& pc)
    {
        std::string x;
        if (!pc.Validate (x)) throw PBException(x);
        MaxNumMetrics = (pc.cache.framesPerTranche / pc.framesPerBlock) + MaxMetricsLatency;
        MaxNumSamples = (pc.cache.framesPerTranche / pc.cache.frameRate) * pc.maxAverageBaseRatePerSecond;

        size_t totalBytes =  sizeof(ZmwResultBuffer) +
                             (MaxMetricsSize * MaxNumMetrics) +
                             (sizeof(TFeat) * MaxNumSamples);

        MaxBufferSize = std::ceil(totalBytes / static_cast<float>(4096)) * 4096;
        MaxNumSamples += (MaxBufferSize - totalBytes) / sizeof(TFeat);
    }

    static size_t MaxNumMetricsPerBuffer()
    {
        return MaxNumMetrics;
    }

    static size_t MaxNumSamplesPerBuffer()
    {
        return MaxNumSamples;
    }

    static size_t SizeOf()
    {
        return MaxBufferSize;
    }

public: // structors

    ZmwResultBuffer() = delete;
    ZmwResultBuffer(const ZmwResultBuffer&) = delete;
    ZmwResultBuffer(ZmwResultBuffer&&) = delete;
    ZmwResultBuffer& operator=(const ZmwResultBuffer&) = delete;
    ZmwResultBuffer& operator=(ZmwResultBuffer&&) = delete;

public: // modifying methods
    void Reset(uint32_t zmwIndex)
    {
        UpdatePointers();

        //if (sentinel_ != 0xABCDEF) PBLOG_WARN << "Warning1: corrupted sentinel:" << std::hex << sentinel_ << std::dec;
        sentinel_     = 0xABCDEF;
        zmwIndex_     = zmwIndex;
        numSamples_   = 0;
        metricIx_     = 0;
        zmwIndexCheck_ = ~zmwIndex;
    }

    /// use this with caution
    void NumSamples(uint32_t count) 
    { numSamples_ = count; }

    TFeat& EditSample(uint32_t offset) 
    { return reinterpret_cast<TFeat*>(samples_)[offset]; }

    template <typename TMetrics>
    TMetrics& EditMetric(uint32_t offset) 
    { return reinterpret_cast<TMetrics*>(metrics_)[offset]; }

    void NumMetrics(uint32_t count) 
    { metricIx_ = count; }

    /// Insert at most count samples.
    size_t BackInsertFeatures(const TFeat* buf, size_t countIn)
    {
        size_t nbytes = std::min<size_t>(end_ - back_, countIn * sizeof(TFeat));
        assert(nbytes % sizeof(TFeat) == 0);

        size_t count = nbytes / sizeof(TFeat);

        std::memcpy(back_, buf, nbytes);
        
        back_ += nbytes;
        numSamples_ += count;

        return count;
    }

    /// Insert at most count metric blocks
    /// 
    /// \param count - contract says, count will always be 0 or 1
    template <typename TMetrics>
    size_t BackInsertMetrics(const TMetrics* buf, size_t count)
    {
        size_t nAvail = MaxNumMetricsPerBuffer() - metricIx_;
        count = std::min(count, nAvail);

        assert(count == 1 || count == 0);

        if (count == 1)
        {
           EditMetric<TMetrics>(metricIx_++) = *buf;
        }

        return count;
    }

public: // non-modifying methods
    size_t Size() const
    {
        // The size is partially fixed and partially variable.
        return sizeof(ZmwResultBuffer) +
               (MaxMetricsSize * MaxNumMetricsPerBuffer()) +
               (sizeof(TFeat) * numSamples_);
    }

    const TFeat* Samples() const
    { return reinterpret_cast<const TFeat*>(samples_); }

    uint32_t NumSamples() const
    { return numSamples_; }

    template <typename TMetrics>
    const TMetrics* Metrics() const
    { return reinterpret_cast<const TMetrics*>(metrics_); }

    uint32_t NumMetrics() const
    { return metricIx_; }

    uint32_t ZmwIndex() const
    { return zmwIndex_; }

     void Check() const
     {
         if (sentinel_ != 0xABCDEF) PBLOG_WARN << "Warning2: corrupted sentinel:" << std::hex << sentinel_ << std::dec;
         if (zmwIndex_ != ~zmwIndexCheck_ ) PBLOG_WARN << "Warning2:corrupted zmw check:" << zmwIndex_ << "!=" <<~zmwIndexCheck_ << ", " << zmwIndexCheck_;
     }

    void StreamOut(std::ostream& os) const
    {
        os << "ZmwIndex:" << ZmwIndex() << " calls[" << numSamples_ <<"]:\n";
        for(uint32_t i=0;i<numSamples_;i++)
        {
            Samples()[i].ToStream(os);
            os << "\n";
        }
    }

protected: // modifying methods

    void UpdatePointers()
    {
        metrics_ = reinterpret_cast<uint8_t*>(this) + sizeof(ZmwResultBuffer);
        samples_ = metrics_ + (MaxMetricsSize * MaxNumMetricsPerBuffer());

        back_ = samples_;
        end_ = back_ + (sizeof(TFeat) * MaxNumSamplesPerBuffer());
    }

private: // static members and methods
    static size_t MaxNumMetrics;
    static size_t MaxNumSamples;
    static size_t MaxBufferSize;

private:

    // Mutable pointer to manage back-insertion operations. These values are
    // not defined after transmission.
    uint8_t* back_;
    const uint8_t* end_;

    uint32_t sentinel_;
    uint32_t zmwIndex_; // zero based index for this acquisition. the number is generated pa-acq by, and then remapped to ZMW Number later.
    uint32_t numSamples_;
    uint32_t metricIx_;
    uint32_t zmwIndexCheck_;

    // Total of 8+8+5*4 = 36 bytes
    uint8_t _padding[28];

    // Start of data:
    // Metrics are reported on regular intervals
    uint8_t* metrics_;

    // Features are variable-length.
    uint8_t* samples_;
};

template <typename TFeat>
size_t ZmwResultBuffer<TFeat>::MaxNumMetrics = 0;

template <typename TFeat>
size_t ZmwResultBuffer<TFeat>::MaxNumSamples = 0;

template <typename TFeat>
size_t ZmwResultBuffer<TFeat>::MaxBufferSize = 0;

using ReadBuffer = ZmwResultBuffer<Basecall>;

static_assert(sizeof(Basecall) <= 32, "sizeof(Basecall) more than 32 bytes");

inline std::ostream& operator<<(std::ostream& s, const ReadBuffer& rb)
{
    rb.StreamOut(s);
    s << std::endl;
    return s;
}

}};
