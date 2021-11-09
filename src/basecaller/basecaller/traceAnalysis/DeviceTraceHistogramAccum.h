// Copyright (c) 2021, Pacific Biosciences of California, Inc.
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

// Several implementations were tried.  While
// SharedInterleaved2DBlock was a clear winner
// and will probably be the only flavor used in
// production, the others are worth keeping around,
// at least until they get in the way.  They may
// prove useful when evaluating new hardware,
// and/or when someone wants to see what optimization
// ides where tried
enum class DeviceHistogramTypes
{
    GlobalInterleaved,
    GlobalContig,
    GlobalContigCoopWarps,
    SharedContig2DBlock,
    SharedContigCoopWarps,
    SharedInterleaved2DBlock
};

class DeviceTraceHistogramAccum : public TraceHistogramAccumulator
{
public:
    static void Configure(const Data::BasecallerTraceHistogramConfig& sigConfig);

    DeviceTraceHistogramAccum(unsigned int poolId,
                             unsigned int poolSize,
                             Cuda::Memory::StashableAllocRegistrar* registrar,
                             DeviceHistogramTypes type = DeviceHistogramTypes::SharedInterleaved2DBlock);

    ~DeviceTraceHistogramAccum();

    void AddBatchImpl(const Data::TraceBatch<DataType>& traces,
                      const TraceHistogramAccumulator::PoolDetModel& detModel,
                      Data::BatchData<DataType>& workspace) override;

    void ResetImpl(const Cuda::Memory::UnifiedCudaArray<LaneHistBounds>& bounds) override;

    void ResetImpl(const Data::BaselinerMetrics& metrics) override;

    PoolHistType HistogramImpl() const override;

    class ImplBase;
private:
    std::unique_ptr<ImplBase> impl_;
};

}}}

#endif //mongo_basecaller_traceAnalysis_DeviceTraceHistogramAccum_H_
