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
#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/MongoConstants.h>
#include <dataTypes/BasecallBatch.h>
#include <dataTypes/LaneDetectionModel.h>
#include <dataTypes/TraceBatch.h>
#include <dataTypes/ConfigForward.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

/// A function-like type that performs trace analysis for a particular batch
/// of ZMWs.
class BatchAnalyzer
{
public:     // Types
    using InputType = PacBio::Mongo::Data::TraceBatch<int16_t>;
    using OutputType = PacBio::Mongo::Data::BasecallBatch;

public:     // Static functions
    /// Sets algorithm configuration and system calibration properties.
    /// Static because the config types keep a JSON representation and
    /// deserialize on each reference, but the values will be the same for
    /// each BatchAnalyzer instance for a given movie.
    /// \note Not thread safe. Do not call this function while threads are
    /// running analysis.
    static void Configure(const Data::BasecallerAlgorithmConfig& bcConfig,
                          const Data::MovieConfig& movConfig);

public:     // Structors & assignment operators
    BatchAnalyzer(uint32_t poolId, const AlgoFactory& algoFac, bool staticAnalysis);

    BatchAnalyzer(const BatchAnalyzer&) = delete;
    BatchAnalyzer(BatchAnalyzer&&);

    BatchAnalyzer& operator=(const BatchAnalyzer&) = delete;
    BatchAnalyzer& operator=(BatchAnalyzer&&) = default;

    ~BatchAnalyzer();

public:
    /// Call operator is non-reentrant and will throw if a trace batch is
    /// received for the wrong ZMW batch or is out of chronological order.
    PacBio::Mongo::Data::BasecallBatch
    operator()(PacBio::Mongo::Data::TraceBatch<int16_t> tbatch);

    PacBio::Mongo::Data::BasecallBatch
    StandardPipeline(PacBio::Mongo::Data::TraceBatch<int16_t> tbatch);
    PacBio::Mongo::Data::BasecallBatch
    StaticModelPipeline(PacBio::Mongo::Data::TraceBatch<int16_t> tbatch);

private:
    uint32_t poolId_;   // ZMW pool being processed by this analyzer.
    uint32_t nextFrameId_ = 0;  // Start frame id expected by the next call.
    std::unique_ptr<Baseliner> baseliner_;
    std::unique_ptr<FrameLabeler> frameLabeler_;
    std::unique_ptr<TraceHistogramAccumulator> traceHistAccum_;
    std::unique_ptr<DetectionModelEstimator> dme_;

    Cuda::Memory::UnifiedCudaArray<Data::LaneModelParameters<Cuda::PBHalf, laneSize>> models_;

    // runs the main compute phases with a static model, bypassing things like the
    // dme and trace binning.  This is necessary for now because they are not
    // even implemented, but may remain desirable in the future when tweaking/profiling
    // steady-state basecalling performance
    bool staticAnalysis_;
};

}}}     // namespace PacBio::Mongo::Basecaller

#endif  // mongo_basecaller_analyzer_BatchAnalyzer_H_
