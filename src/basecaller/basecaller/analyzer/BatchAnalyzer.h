#ifndef mongo_basecaller_analyzer_BatchAnalyzer_H_
#define mongo_basecaller_analyzer_BatchAnalyzer_H_

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
//  Description:
//  Defines class BatchAnalyzer.

#include <memory>

#include <basecaller/traceAnalysis/TraceAnalysisForward.h>
#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>
#include <dataTypes/BatchResult.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/PulseBatch.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A function-like type that performs trace analysis for a particular batch
/// of ZMWs.
class BatchAnalyzer
{
public:     // Types
    using InputType = PacBio::Mongo::Data::TraceBatch<int16_t>;
    using OutputType = PacBio::Mongo::Data::BatchResult;

public:
    static void ReportPerformance();

public:     // Structors & assignment operators
    BatchAnalyzer(uint32_t poolId,
                  const Data::BatchDimensions& dims,
                  const AlgoFactory& algoFac,
                  Cuda::Memory::DeviceAllocationStash& stash);

    BatchAnalyzer(const BatchAnalyzer&) = delete;
    BatchAnalyzer(BatchAnalyzer&&) = default;

    BatchAnalyzer& operator=(const BatchAnalyzer&) = delete;
    BatchAnalyzer& operator=(BatchAnalyzer&&);

    ~BatchAnalyzer();

public:
    /// Call operator is non-reentrant and will throw if a trace batch is
    /// received for the wrong ZMW batch or is out of chronological order.
    // TODO clean this change to reference
    OutputType operator()(const PacBio::Mongo::Data::TraceBatch<int16_t>& tbatch);

    uint32_t PoolId() const { return poolId_; }

    // TODO these really should be private, with protected accessors.  Children should
    //      be able to use these and even call non-const methods, but should not be able
    //      to destroy the contained type.
protected:
    std::unique_ptr<Baseliner> baseliner_;
    std::unique_ptr<FrameLabeler> frameLabeler_;
    std::unique_ptr<PulseAccumulator> pulseAccumulator_;
    std::unique_ptr<DetectionModelEstimator> dme_;
    std::unique_ptr<HFMetricsFilter> hfMetrics_;

    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>> models_;

private:
    virtual OutputType AnalyzeImpl(const PacBio::Mongo::Data::TraceBatch<int16_t>& tbatch) = 0;

    uint32_t poolId_;   // ZMW pool being processed by this analyzer.
    int32_t nextFrameId_ = 0;  // Start frame id expected by the next call.
};

// Does a single dme estimate upfront, once we've gathered enough
// data in the baseline stats and histogram
class SingleEstimateBatchAnalyzer : public BatchAnalyzer
{
public:
    SingleEstimateBatchAnalyzer(uint32_t poolId,
                                const Data::BatchDimensions& dims,
                                const AlgoFactory& algoFac,
                                Cuda::Memory::DeviceAllocationStash& stash);

    SingleEstimateBatchAnalyzer(const SingleEstimateBatchAnalyzer&) = delete;
    SingleEstimateBatchAnalyzer(SingleEstimateBatchAnalyzer&&) = default;

    SingleEstimateBatchAnalyzer& operator=(const SingleEstimateBatchAnalyzer&) = delete;
    SingleEstimateBatchAnalyzer& operator=(SingleEstimateBatchAnalyzer&&) = default;

    ~SingleEstimateBatchAnalyzer() = default;

    OutputType AnalyzeImpl(const PacBio::Mongo::Data::TraceBatch<int16_t>& tbatch) override;
private:
    bool isModelInitialized_ {false};
    // Marks the first frame that get's a real model estimate and
    // can produce legitimate basecalls.  This variable is meaningless
    // while isModelInitialized == false.
    uint32_t firstFrameWithEstimates_ = std::numeric_limits<uint32_t>::max();

};

// Runs a version of the pipeline with no DME stage.  The model is setallmeans
// statically up-front to match the (simulated) input data
class FixedModelBatchAnalyzer : public BatchAnalyzer
{
public:
    FixedModelBatchAnalyzer(uint32_t poolId,
                            const Data::BatchDimensions& dims,
                            const Data::StaticDetModelConfig& staticDetModelConfig,
                            const Data::MovieConfig& movieConfig,
                            const AlgoFactory& algoFac,
                            Cuda::Memory::DeviceAllocationStash& stash);

    FixedModelBatchAnalyzer(const FixedModelBatchAnalyzer&) = delete;
    FixedModelBatchAnalyzer(FixedModelBatchAnalyzer&&) = default;

    FixedModelBatchAnalyzer& operator=(const FixedModelBatchAnalyzer&) = delete;
    FixedModelBatchAnalyzer& operator=(FixedModelBatchAnalyzer&&) = default;

    ~FixedModelBatchAnalyzer() = default;

    OutputType AnalyzeImpl(const PacBio::Mongo::Data::TraceBatch<int16_t>& tbatch) override;
};

// Runs continuous (staggered) dme estimations that update as new data comes in.
// Estimations are evently staggered between different pools, so that chunks are
// all roughly the same computational expense
class DynamicEstimateBatchAnalyzer : public BatchAnalyzer
{
public:
    DynamicEstimateBatchAnalyzer(uint32_t poolId,
                                 uint32_t maxPoolId,
                                 const Data::BatchDimensions& dims,
                                 const Data::BasecallerDmeConfig& dmeConfig,
                                 const AlgoFactory& algoFac,
                                 Cuda::Memory::DeviceAllocationStash& stash);

    DynamicEstimateBatchAnalyzer(const DynamicEstimateBatchAnalyzer&) = delete;
    DynamicEstimateBatchAnalyzer(DynamicEstimateBatchAnalyzer&&) = default;

    DynamicEstimateBatchAnalyzer& operator=(const DynamicEstimateBatchAnalyzer&) = delete;
    DynamicEstimateBatchAnalyzer& operator=(DynamicEstimateBatchAnalyzer&&) = default;

    ~DynamicEstimateBatchAnalyzer() = default;

    OutputType AnalyzeImpl(const PacBio::Mongo::Data::TraceBatch<int16_t>& tbatch) override;
private:
    uint32_t poolDmeDelayFrames_;
    bool fullEstimationOccured_ {false};
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif  // mongo_basecaller_analyzer_BatchAnalyzer_H_
