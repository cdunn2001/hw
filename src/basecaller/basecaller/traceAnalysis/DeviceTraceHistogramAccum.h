// Copyright (c) 2019, Pacific Biosciences of California, Inc.
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

#ifndef mongo_basecaller_traceAnalysis_DeviceTraceHistogramAccum_H_
#define mongo_basecaller_traceAnalysis_DeviceTraceHistogramAccum_H_

#include <basecaller/traceAnalysis/TraceHistogramAccumulator.h>

#include <common/cuda/memory/DeviceAllocationStash.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

enum class DeviceHistogramTypes
{
    GlobalInterleaved,
    GlobalInterleavedMulti,
    GlobalContig,
    GlobalCotigMulti,
    GlobalContigAtomic,
    SharedContigMulti,
    SharedContigAtomic,
    SharedInterleaved
};

class DeviceTraceHistogramAccumHidden : public TraceHistogramAccumulator
{
public:
    static void Configure(const Data::BasecallerTraceHistogramConfig& sigConfig);

    DeviceTraceHistogramAccumHidden(unsigned int poolId,
                                    unsigned int poolSize,
                                    DeviceHistogramTypes type,
                                    Cuda::Memory::StashableAllocRegistrar* registrar);

    ~DeviceTraceHistogramAccumHidden();

    void AddBatchImpl(const Data::TraceBatch<DataType>& traces) override;

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override;

    void ResetImpl(const Data::BaselinerMetrics& metrics) override;

    PoolHistType HistogramImpl() const override;

    struct ImplBase;
private:
    std::unique_ptr<ImplBase> impl_;
};

template <DeviceHistogramTypes type>
class DeviceTraceHistogramAccum : public DeviceTraceHistogramAccumHidden
{
public:
    DeviceTraceHistogramAccum(unsigned int poolId,
                              unsigned int poolSize,
                              Cuda::Memory::StashableAllocRegistrar* registrar)
        : DeviceTraceHistogramAccumHidden(poolId, poolSize, type, registrar)
    {}

    using DeviceTraceHistogramAccumHidden::ResetImpl;
    using DeviceTraceHistogramAccumHidden::HistogramImpl;

};


}}}

#endif //mongo_basecaller_traceAnalysis_DeviceTraceHistogramAccum_H_
