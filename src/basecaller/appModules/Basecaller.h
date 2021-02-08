// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

#include "pacbio/dev/AutoTimer.h"
#include <map>
#include <vector>

#include <common/cuda/memory/DeviceAllocationStash.h>
#include <common/cuda/streams/CudaStream.h>
#include <common/graphs/GraphNodeBody.h>

#include <pacbio/ipc/ThreadSafeQueue.h>

#include <basecaller/analyzer/AlgoFactory.h>
#include <basecaller/analyzer/BatchAnalyzer.h>
#include <basecaller/traceAnalysis/AnalysisProfiler.h>
#include <basecaller/traceAnalysis/ComputeDevices.h>

#include <dataTypes/BatchResult.h>
#include <dataTypes/configs/BasecallerAlgorithmConfig.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/configs/SystemsConfig.h>
#include <dataTypes/TraceBatch.h>

namespace PacBio {
namespace Application {

class BasecallerBody final : public Graphs::TransformBody<const Mongo::Data::TraceBatch<int16_t>, Mongo::Data::BatchResult>
{
    using BatchAnalyzer = Mongo::Basecaller::BatchAnalyzer;
public:
    BasecallerBody(const std::map<uint32_t, Mongo::Data::BatchDimensions>& poolDims,
                   const Mongo::Data::BasecallerAlgorithmConfig& algoConfig,
                   const Mongo::Data::MovieConfig& movConfig,
                   const Mongo::Data::SystemsConfig& sysConfig,
                   size_t maxPermGpuMemSize = std::numeric_limits<size_t>::max())
        : gpuStash(std::make_unique<Cuda::Memory::DeviceAllocationStash>())
        , algoFactory_(algoConfig)
        , streams_(std::make_unique<PacBio::ThreadSafeQueue<std::unique_ptr<Cuda::CudaStream>>>())
        , measurePCIeBandwidth_(sysConfig.analyzerHardware != Mongo::Basecaller::ComputeDevices::Host)
        , numStreams_(sysConfig.basecallerConcurrency)
    {
        auto priorityRange = Cuda::StreamPriorityRange();
        assert(priorityRange.greatestPriority < priorityRange.leastPriority);
        PBLOG_INFO << "Assigning priority range " << priorityRange.greatestPriority << ":" << priorityRange.leastPriority
                   << " round robbin amongst " << numStreams_ << " streams";
        auto priority = priorityRange.greatestPriority;
        for (size_t i = 0; i < numStreams_; ++i)
        {
            streams_->Push(std::make_unique<Cuda::CudaStream>(priority));
            priority++;
            if (priority > priorityRange.leastPriority) priority = priorityRange.greatestPriority;
        }
        using namespace Mongo::Data;

        algoFactory_.Configure(algoConfig, movConfig);

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
                                                                     movConfig, algoFactory_,
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
        gpuStash->PartitionData(maxPermGpuMemSize);
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

    size_t ConcurrencyLimit() const override { return numStreams_; }
    float MaxDutyCycle() const override { return .8; }

    Mongo::Data::BatchResult Process(const Mongo::Data::TraceBatch<int16_t>& in) override
    {
        using IOProfiler = Mongo::Basecaller::IOProfiler;
        using IOStages = Mongo::Basecaller::IOStages;
        IOProfiler profiler(IOProfiler::Mode::OBSERVE, 100, 100);

        auto lockProfiler = profiler.CreateScopedProfiler(IOStages::Locks);
        (void) lockProfiler;

        {
            static std::mutex reportMutex;
            std::lock_guard<std::mutex> lm(reportMutex);
            if (in.Metadata().FirstFrame() < currFrame_)
                throw PBException("Received an out-of-order batch");
            if (in.Metadata().FirstFrame() > currFrame_ && measurePCIeBandwidth_)
            {
                auto gbUpload = bytesUploaded_ >> 30;
                auto secUpload = msUpload / 1000;
                PBLOG_INFO << "Uploaded " << gbUpload
                           << "GiB to GPU in " << secUpload << "s (" << gbUpload/secUpload << "GiB/s)";
                auto gbDownload = bytesDownloaded_ >> 30;
                auto secDownload = msDownload / 1000;
                PBLOG_INFO << "Downloaded " << gbDownload
                           << "GiB from GPU in " << secDownload << "s (" << gbDownload/secDownload << "GiB/s)";

                currFrame_ = in.Metadata().FirstFrame();
                bytesUploaded_ = 0;
                bytesDownloaded_ = 0;
                msUpload = 0.0;
                msDownload = 0.0;
            }
        }

        auto stream = streams_->Pop();
        auto f1 = stream->SetAsDefaultStream();
        Utilities::Finally f2([&, this](){
            streams_->Push(std::move(stream));
        });
        auto& analyzer = *bAnalyzer_.at(in.GetMeta().PoolId());

        {
            static std::mutex uploadMutex;
            std::lock_guard<std::mutex> lm(uploadMutex);

            auto uploadProfiler = profiler.CreateScopedProfiler(IOStages::Upload);
            (void) uploadProfiler;

            PacBio::Dev::Profile::FastTimer timer;
            if (measurePCIeBandwidth_)
                bytesUploaded_ += in.CopyToDevice();
            bytesUploaded_ += gpuStash->RetrievePool(in.GetMeta().PoolId());
            msUpload += timer.GetElapsedMilliseconds();
        }

        auto ret = [&](){
            auto computeProfiler = profiler.CreateScopedProfiler(IOStages::Compute);
            (void) computeProfiler;
            return analyzer(std::move(in));
        }();

        {
            static std::mutex downloadMutex;
            std::lock_guard<std::mutex> lm(downloadMutex);

            auto downloadProfiler = profiler.CreateScopedProfiler(IOStages::Download);
            (void) downloadProfiler;

            PacBio::Dev::Profile::FastTimer timer;
            if (measurePCIeBandwidth_)
                bytesDownloaded_ += ret.DeactivateGpuMem();
            bytesDownloaded_ += gpuStash->StashPool(in.GetMeta().PoolId());
            msDownload += timer.GetElapsedMilliseconds();
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

    uint32_t currFrame_ = 0;
    uint32_t numStreams_;
    size_t bytesUploaded_ = 0;
    size_t bytesDownloaded_ = 0;
    double msUpload = 0.0;
    double msDownload = 0.0;
};

}}

#endif //PACBIO_APPLICATION_ANALYZER_H
