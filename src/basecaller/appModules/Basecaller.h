// Copyright (c) 2020-2021, Pacific Biosciences of California, Inc.
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

#ifndef PACBIO_APPLICATION_ANALYZER_H
#define PACBIO_APPLICATION_ANALYZER_H

#include <map>
#include <vector>

#include <pacbio/ipc/ThreadSafeQueue.h>

#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/streams/CudaStream.h>
#include <common/graphs/GraphNodeBody.h>

#include <basecaller/analyzer/AlgoFactory.h>
#include <basecaller/analyzer/BatchAnalyzer.h>
#include <basecaller/traceAnalysis/AnalysisProfiler.h>
#include <basecaller/traceAnalysis/ComputeDevices.h>

#include <dataTypes/BatchResult.h>
#include <dataTypes/configs/BasecallerAlgorithmConfig.h>
#include <dataTypes/configs/AnalysisConfig.h>
#include <dataTypes/configs/SystemsConfig.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Application {

class BasecallerBody final : public Graphs::TransformBody<const Mongo::Data::TraceBatchVariant, Mongo::Data::BatchResult>
{
    using BatchAnalyzer = Mongo::Basecaller::BatchAnalyzer;
public:
    BasecallerBody(const std::map<uint32_t, Mongo::Data::BatchDimensions>& poolDims,
                   const Mongo::Data::BasecallerAlgorithmConfig& algoConfig,
                   const Mongo::Data::AnalysisConfig& analysisConfig,
                   const Mongo::Data::SystemsConfig& sysConfig)
        : gpuStash(std::make_unique<Cuda::Memory::DeviceAllocationStash>())
        , algoFactory_(algoConfig)
        , streams_(std::make_unique<PacBio::ThreadSafeQueue<std::unique_ptr<Cuda::CudaStream>>>())
        , measurePCIeBandwidth_(algoConfig.ComputingMode() == Mongo::Data::BasecallerAlgorithmConfig::ComputeMode::PureGPU)
        , usesGpu_(algoConfig.ComputingMode() != Mongo::Data::BasecallerAlgorithmConfig::ComputeMode::PureHost)
        , basecallerConcurrency_(sysConfig.basecallerConcurrency)
    {
        if (usesGpu_)
        {
            auto priorityRange = Cuda::StreamPriorityRange();
            assert(priorityRange.greatestPriority <= priorityRange.leastPriority);
            PBLOG_INFO << "Assigning priority range " << priorityRange.greatestPriority << ":" << priorityRange.leastPriority
                       << " round robbin amongst " << basecallerConcurrency_ << " streams";
            auto priority = priorityRange.greatestPriority;

            for (size_t i = 0; i < basecallerConcurrency_; ++i)
            {
                streams_->Push(std::make_unique<Cuda::CudaStream>(priority));
                priority++;
                if (priority > priorityRange.leastPriority) priority = priorityRange.greatestPriority;
            }
        }
        using namespace Mongo::Data;

        algoFactory_.Configure(algoConfig, analysisConfig);

        // TODO this computation will not be sufficient for sparse layouts
        uint32_t maxPoolId = std::accumulate(poolDims.begin(), poolDims.end(), 0u,
                                             [](uint32_t currMax, auto&& kv)
                                             {
                                                 return std::max(currMax, kv.first);
                                             });

        for (const auto & kv : poolDims)
        {
            const auto& poolId = kv.first;
            const auto& dims = kv.second;
            const auto it = bAnalyzer_.find(poolId);
            if (it != bAnalyzer_.cend()) continue;  // Ignore duplicate ids.

            using namespace Mongo::Basecaller;
            auto batchAnalyzer = [&]() -> std::unique_ptr<BatchAnalyzer>
            {
                switch (algoConfig.modelEstimationMode)
                {
                case BasecallerAlgorithmConfig::ModelEstimationMode::FixedEstimations:
                    return std::make_unique<FixedModelBatchAnalyzer>(poolId, dims,
                                                                     algoConfig.staticDetModelConfig,
                                                                     analysisConfig, algoFactory_,
                                                                     *gpuStash);
                case BasecallerAlgorithmConfig::ModelEstimationMode::InitialEstimations:
                    return std::make_unique<SingleEstimateBatchAnalyzer>(poolId, dims, algoFactory_,
                                                                         *gpuStash);
                case BasecallerAlgorithmConfig::ModelEstimationMode::DynamicEstimations:
                    return std::make_unique<DynamicEstimateBatchAnalyzer>(poolId,
                                                                          maxPoolId,
                                                                          dims,
                                                                          algoConfig.dmeConfig,
                                                                          algoFactory_,
                                                                          *gpuStash);
                default:
                    throw PBException("Unexpected model estimation mode: "
                                     + algoConfig.modelEstimationMode.toString());
                }
            }();

            bAnalyzer_.emplace(poolId, std::move(batchAnalyzer));
        }

        if (usesGpu_)
            gpuStash->PartitionData(sysConfig.maxPermGpuDataMB);
    }

    BasecallerBody(const BasecallerBody&) = delete;
    BasecallerBody(BasecallerBody&&) = default;
    BasecallerBody& operator=(const BasecallerBody&) = delete;
    BasecallerBody& operator=(BasecallerBody&&) = default;

    ~BasecallerBody()
    {
        BatchAnalyzer::ReportPerformance();
        Mongo::Basecaller::IOProfiler::FinalReport();
    }

    size_t ConcurrencyLimit() const override { return basecallerConcurrency_; }
    float MaxDutyCycle() const override { return .8; }

    // Notes:
    //   There are a couple work scheduling concerns that the below code is being
    //   a touch careful about.  In general, most of what you need is taken care of
    //   by throwing 3+ cuda streams at the problem, and things will automatically tend
    //   towards one stream uploading data, one stream doing compute, another stream
    //   downloading (and any extra streams filling in scheduling gaps).
    //
    //   That said, the simple thing only works robustly when there is a single data
    //   upload/download and a single kernel invocation.  When there are multiple pieces
    //   of data to upload, you can occasionally work yourself into a situation where every
    //   stream is taking turns uploading data and no one is overlapping that with compute.
    //   Similarly you can get multiple streams fighting over the compute resources with
    //   no one overlapping that with IO.  This introduces "bubbles" in the work scheduling
    //   that make our overal compute more inefficient than otherwise possible.
    //
    //   To combat this, the below code does two things:
    //     * The main upload and download are hidden behind a lock and mutex, reducing contention
    //       over IO resources and allowing a thread to finish it's data movement as fast as
    //       possible so it can proceed to compute
    //     * The streams have been given a "priority", which can serve as a sort of tie breaker
    //       when multiple streams are doing simultaneous compute.  We actually do want some
    //       ammount of compute overlap, but using priorities helps ensure one thread is working as
    //       fast as possible and getting to it's next IO stage quickly, and the rest are relegated
    //       to "filling in the gaps" so to speak.
    //
    //   Final note: There are not that many unique stream priorities.  On the A100 and V100
    //               there are only three.  As we expect 4-5 streams to be in play, this
    //               mechanism only serves as an imperfect tie breaker.  For now this is OK
    //               as the IO mutexes on their own are enough to mostly fix the problem.  If
    //               that ever proves insuficcient we can have dedicated upload/download streams
    //               and only use the priority streams for compute, which would allow us to get to
    //               5 total active streams without reusing priority values.  The main reason
    //               this wasn't done this round is that approach does make the timelines a bit
    //               harder to parse in the cuda profilers.
    Mongo::Data::BatchResult Process(const Mongo::Data::TraceBatchVariant& in) override
    {
        using IOProfiler = Mongo::Basecaller::IOProfiler;
        using IOStages = Mongo::Basecaller::IOStages;
        IOProfiler profiler(IOProfiler::Mode::OBSERVE, 100, 100);

        // top level profiler to keep an eye on how long we are stuck in
        // overhead things like lock acquiring
        auto lockProfiler = profiler.CreateScopedProfiler(IOStages::Locks);
        (void) lockProfiler;

        {
            static std::mutex reportMutex;
            std::lock_guard<std::mutex> lm(reportMutex);

            if (in.Metadata().FirstFrame() < currFrame_)
                throw PBException("Received an out-of-order batch");
            if (in.Metadata().FirstFrame() > currFrame_ && measurePCIeBandwidth_)
            {
                // These shouldn't be able to trigger unless there is an outright bug
                // in the timing code. These variables are in units of miliseconds, but
                // have double precision and accumulate data from full resolution timers.
                // Just setup/teardown should be enough to give us a nonzero interval even
                // if for some reason we transfer no data.
                assert(msUpload > 0);
                assert(msDownload > 0);

                auto gbUpload = bytesUploaded_ >> 30;
                auto secUpload = msUpload / 1000.;
                PBLOG_INFO << "Uploaded " << gbUpload
                           << "GiB to GPU in " << secUpload << "s (" << gbUpload/secUpload << "GiB/s)";
                auto gbDownload = bytesDownloaded_ >> 30;
                auto secDownload = msDownload / 1000.;
                PBLOG_INFO << "Downloaded " << gbDownload
                           << "GiB from GPU in " << secDownload << "s (" << gbDownload/secDownload << "GiB/s)";

                currFrame_ = in.Metadata().FirstFrame();
                bytesUploaded_ = 0;
                bytesDownloaded_ = 0;
                msUpload = 0.0;
                msDownload = 0.0;
            }
        }

        // Grab a cuda stream from the queue.  We'll set up
        // a couple "Finally" statements to make sure we
        // robustly undo our change to the default stream and
        // put the stream back in the queue, even if there
        // is an exception
        auto threadStream = usesGpu_ ? streams_->Pop() : nullptr;
        Utilities::Finally finally1([&, this]() {
            if (threadStream) streams_->Push(std::move(threadStream));
        });
        auto finally2 = threadStream ? threadStream->SetAsDefaultStream() : []() {};

        auto& analyzer = *bAnalyzer_.at(in.Metadata().PoolId());

        {
            static std::mutex uploadMutex;
            std::lock_guard<std::mutex> lm(uploadMutex);

            auto uploadProfiler = profiler.CreateScopedProfiler(IOStages::Upload);
            (void) uploadProfiler;

            PacBio::Dev::Profile::FastTimer timer;
            if (measurePCIeBandwidth_)
            {
                bytesUploaded_ += std::visit([](const auto& batch)
                {
                    return batch.CopyToDevice();
                }, in.Data());
                Cuda::CudaSynchronizeDefaultStream();
            }
            if (usesGpu_)
            {
                bytesUploaded_ += gpuStash->RetrievePool(in.Metadata().PoolId());
                msUpload += timer.GetElapsedMilliseconds();
            }
        }

        auto ret = [&](){
            auto computeProfiler = profiler.CreateScopedProfiler(IOStages::Compute);
            (void) computeProfiler;
            return analyzer(in);
        }();

        {
            static std::mutex downloadMutex;
            std::lock_guard<std::mutex> lm(downloadMutex);

            auto downloadProfiler = profiler.CreateScopedProfiler(IOStages::Download);
            (void) downloadProfiler;

            if (usesGpu_)
            {
                PacBio::Dev::Profile::FastTimer timer;
                bytesDownloaded_ += ret.DeactivateGpuMem();
                bytesDownloaded_ += gpuStash->StashPool(in.Metadata().PoolId());
                msDownload += timer.GetElapsedMilliseconds();
            }
        }
        return ret;
    }
private:

    std::unique_ptr<Cuda::Memory::DeviceAllocationStash> gpuStash;
    Mongo::Basecaller::AlgoFactory algoFactory_;

    // One analyzer for each pool. Key is pool id.
    std::map<uint32_t, std::unique_ptr<BatchAnalyzer>> bAnalyzer_;

    std::unique_ptr<PacBio::ThreadSafeQueue<std::unique_ptr<Cuda::CudaStream>>> streams_;

    bool measurePCIeBandwidth_;
    bool usesGpu_;

    int32_t currFrame_ = 0;
    uint32_t basecallerConcurrency_;
    size_t bytesUploaded_ = 0;
    size_t bytesDownloaded_ = 0;
    double msUpload = 0.0;
    double msDownload = 0.0;
};

}}

#endif //PACBIO_APPLICATION_ANALYZER_H
