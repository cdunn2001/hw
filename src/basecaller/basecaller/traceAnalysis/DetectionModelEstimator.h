#ifndef mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
#define mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_

// Copyright (c) 2019,2020 Pacific Biosciences of California, Inc.
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
//  Defines class DetectionModelEstimator.

#include <basecaller/traceAnalysis/TraceAnalysisForward.h>
#include <basecaller/traceAnalysis/AnalysisProfiler.h>

#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/PBCudaSimd.h>

#include <dataTypes/BatchMetrics.h>
#include <dataTypes/configs/ConfigForward.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/LaneDetectionModel.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

// Top level class that coordinates detection model estimation.
// Most of what it does is bookkeeping, using it's
// constituent CoreDMEstimator, TraceHistogramAccumulator and
// BaselineStatsAggregator to perform the actual computations
class DetectionModelEstimator
{
public: // static functions
    static void Configure(const Data::BasecallerDmeConfig& dmeConfig);

    /// Minimum number of frames in baseline stats aggregation before
    /// estimating histogram bounds
    static unsigned int NumFramesPreAccumStats()
    { return numFramesPreAccumStats_; }

    /// Minimum number of frames added to trace histograms before we estimate
    /// model parameters.
    static uint32_t MinFramesForEstimate()
    { return minFramesForEstimate_; }


public:
    using LaneDetModel = Data::LaneDetectionModel<Cuda::PBHalf>;
    using PoolDetModel = Data::DetectionModelPool<Cuda::PBHalf>;

    DetectionModelEstimator(uint32_t poolId,
                            const Data::BatchDimensions& dims,
                            Cuda::Memory::StashableAllocRegistrar& registrar,
                            const AlgoFactory& algoFac);


    ~DetectionModelEstimator();

    // Returns true if adding this batch caused the dme to run a full estimation
    // attempt and possibly update the models.
    // Note: The models can still be updated even if there was not a full
    //       estimation performed
    bool AddBatch(const Data::TraceBatch<int16_t>& traces,
                  const Data::BaselinerMetrics& metrics,
                  PoolDetModel* models,
                  AnalysisProfiler& profiler);

    // Number of frames before the first full estimation attempt
    uint32_t StartupLatency() const;

private:    // Static data
    // Number of frames to accumulate baseliner statistics before initializing
    // histograms.
    static uint32_t numFramesPreAccumStats_;
    // Number of frames between runs of coreEstimator
    static uint32_t minFramesForEstimate_;

private:

    enum class PoolStatus
    {
        STARTUP_HIST_INIT,  // Gathering of baseline stats leading towards intial historam setup
        STARTUP_DME_INIT,   // Gathering of the first histogram leading toward initial estimation
        SEQUENCING,         // Producing potentially useful results
    };
    PoolStatus poolStatus_ {PoolStatus::STARTUP_HIST_INIT};
    // Number of frames before the next event (e.g. state machine transition, or full estimation)
    int32_t framesRemaining_;
    uint32_t framesPerBatch_;

    // Actual worker classes.  Nothing in DetectionModelEstimator should actually examine lane
    // level data, that should all be done by these classes, which potentially can have either
    // CPU or GPU implementations
    std::unique_ptr<TraceHistogramAccumulator> traceAccumulator_;
    std::unique_ptr<BaselineStatsAggregator> baselineAggregator_;
    std::unique_ptr<CoreDMEstimator> coreEstimator_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif // mongo_basecaller_traceAnalysis_DetectionModelEstimator_H_
