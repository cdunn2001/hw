#include <common/ZmwDataManager.h>

#include <common/cuda/PBCudaSimd.h>

#include <pacbio/PBException.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/utilities/SmartEnum.h>
#include <pacbio/dev/profile/ScopedProfilerChain.h>

namespace PacBio {
namespace Cuda {
namespace Data {

SMART_ENUM(DATA, FILL_BANK, FILL_KERNEL);

template <typename TIn, typename TOut>
ZmwDataManager<TIn, TOut>::ZmwDataManager(
    const DataManagerParams& params,
    std::unique_ptr<GeneratorBase<TIn>> source,
    bool quiet)
        : params_(params)
        , exitRequested_(false)
        , running_(true)
        , fillingBank_(false)
        , batchesReady_(0)
        , batchesLoaned_(0)
        , computeBlock_(0)
        , quiet_(quiet)
        , source_(std::move(source))
{

    // enforce even multiples for simplicity...
    if (params_.numZmwLanes % params_.kernelLanes != 0)
        throw PBException("Select numZmw that is even multiple of kernelLanes");
    // A batch is a group of lanes that will be sent to the GPU together
    numBatches_ = params_.numZmwLanes / params_.kernelLanes;
    const size_t count = params_.kernelLanes * params_.laneWidth * params_.blockLength;

    Mongo::Data::BatchDimensions batchDims;
    batchDims.laneWidth = params_.laneWidth;
    batchDims.framesPerBatch = params_.blockLength;
    batchDims.lanesPerBatch = params_.kernelLanes;

    poolIn_ = std::make_shared<Memory::DualAllocationPools>(count*sizeof(TIn));
    poolOut_ = std::make_shared<Memory::DualAllocationPools>(count*sizeof(TOut));
    for (size_t i = 0; i < numBatches_; ++i)
    {
        using Mongo::Data::BatchMetadata;
        bank0.emplace_back(BatchMetadata(i, 0, params_.blockLength),
                           batchDims,
                           Memory::SyncDirection::HostWriteDeviceRead,
                           poolIn_);
        bank1.emplace_back(BatchMetadata(i, 0, params_.blockLength),
                           batchDims,
                           Memory::SyncDirection::HostWriteDeviceRead,
                           poolIn_);
        output.emplace_back(BatchMetadata(i, 0, params_.blockLength),
                            batchDims,
                            Memory::SyncDirection::HostReadDeviceWrite,
                            poolOut_);
    }
    checkedOut_.resize(numBatches_, false);

    thread_ = std::thread(&ZmwDataManager::FillerThread, this);
}

template <typename TIn, typename TOut>
auto ZmwDataManager<TIn, TOut>::NextBatch() -> LoanedData
{
    std::lock_guard<std::mutex> l(nextBatchMutex_);
    while (batchesReady_ == 0)
    {
        if (!MoreData()) throw PBException("No data available for NextBatch() to return");
    }

    auto retIdx = numBatches_ - batchesReady_;
    if (checkedOut_[retIdx] != 0)
    {
        std::cerr << "error checking out index " << retIdx << std::endl;
        throw PBException("Trying to check grab data already checked out!");
    }
    checkedOut_[retIdx] = 1;
    batchesLoaned_++;
    batchesReady_--;
    auto& bank = fillingBank_ ? bank1 : bank0;
    output[retIdx].SetMeta(bank[retIdx].GetMeta());
    return LoanedData(std::move(bank[retIdx]), std::move(output[retIdx]));
}

template <typename TIn, typename TOut>
void ZmwDataManager<TIn, TOut>::ReturnBatch(LoanedData&& data)
{
    if (checkedOut_.at(data.Batch()) == 0)
    {
        PBLOG_ERROR << "Error returning index " << data.Batch();
        throw PBException("Trying to return data that is not checked out!");
    }
    data.KernelInput().DeactivateGpuMem();
    data.KernelOutput().DeactivateGpuMem();

    auto& bank = fillingBank_ ? bank1 : bank0;
    auto batch = data.Batch();
    bank[batch] = std::move(data.KernelInput());
    output[batch] = std::move(data.KernelOutput());

    checkedOut_[batch] = 0;
    if (batchesLoaned_ == 1 && batchesReady_ == 0 && !quiet_)
        PBLOG_INFO << "Compute time: " << computeTimer_.GetElapsedMilliseconds();
    batchesLoaned_--;
}

template <typename TIn, typename TOut>
ZmwDataManager<TIn, TOut>::~ZmwDataManager()
{
    exitRequested_ = true;
    if (thread_.joinable()) thread_.join();
    for (auto val : checkedOut_)
    {
        if(val) PBLOG_ERROR << "Destroying ZmwDataManager while data still checked out! There are likely dangling references in the wild";
    }
}

template <typename TIn, typename TOut>
void ZmwDataManager<TIn, TOut>::FillNext(Mongo::Data::TraceBatch<TIn>& ptr)
{
    for (size_t i = 0; i < ptr.LanesPerBatch(); ++i)
    {
        source_->Fill(laneIdx_, blockIdx_, ptr.GetBlockView(i));
        laneIdx_++;
        if (laneIdx_ == params_.numZmwLanes)
        {
            laneIdx_ = 0;
            blockIdx_++;
        }
        if (blockIdx_ == params_.numBlocks && i < ptr.LanesPerBatch()-1)
            throw PBException("ZmwDataManager can only fill integral batches");
    }
    if (params_.immediateCopy)
    {
        ptr.CopyToDevice();
    }
}

template <typename TIn, typename TOut>
void ZmwDataManager<TIn, TOut>::FillerThread()
{
    const float requestedDurationMs = params_.blockLength  / params_.frameRate * 1e3f;
    using Profiler = PacBio::Dev::Profile::ScopedProfilerChain<DATA>;

    auto fill = [&](std::vector<Mongo::Data::TraceBatch<TIn>>& bank) {
        auto mode = blockIdx_ < 3 ? Profiler::Mode::OBSERVE : Profiler::Mode::REPORT;
        Profiler prof(mode, 1.0f, 9e9);
        auto bankProf = prof.CreateScopedProfiler(DATA::FILL_BANK);
        (void)bankProf;
        if (laneIdx_ != 0) throw PBException("Implementation error");
        const size_t startBlock = blockIdx_;
        size_t batchId = 0;
        const size_t startFrame = blockIdx_ * params_.blockLength;
        const size_t stopFrame = startFrame + params_.blockLength;
        for (auto& b : bank)
        {
            auto kernelProf = prof.CreateScopedProfiler(DATA::FILL_KERNEL);
            (void)kernelProf;

            Mongo::Data::BatchMetadata meta(batchId, startFrame, stopFrame);
            b.SetMeta(meta);
            FillNext(b);

            batchId++;
        }
        if (laneIdx_ != 0) throw PBException("implementation error");
        if (blockIdx_ != startBlock + 1) throw PBException("implementation error");
    };

    for (size_t i = 0; i < params_.numBlocks; ++i)
    {
        if (exitRequested_) return;

        Dev::QuietAutoTimer timer;
        if (!quiet_)
            PBLOG_INFO << "Filling Bank " << i;
        if (fillingBank_)
            fill(bank0);
        else
            fill(bank1);

        size_t elapsed = timer.GetElapsedMilliseconds();
        if (elapsed > requestedDurationMs)
        {
            PBLOG_WARN << "Filling thread not able to sustain requested framerate! "
                       << elapsed << " " << requestedDurationMs;
        } else {
            auto sleepDuration = static_cast<size_t>(requestedDurationMs - elapsed);
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepDuration));
        }

        while (batchesReady_ > 0 || batchesLoaned_ > 0)
        {
            PBLOG_WARN << "Compute was not fast enough, IO thread sleeping!!\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        computeBlock_ = blockIdx_-1;
        computeTimer_.Restart();
        fillingBank_ = !fillingBank_;
        batchesReady_ = numBatches_;
    }
    PBLOG_INFO << "IO thread complete, waiting for termination\n";
    running_ = false;

    if (!quiet_)
        Profiler::FinalReport();
}

template class ZmwDataManager<short>;
template class ZmwDataManager<PBShort2>;

}}}
