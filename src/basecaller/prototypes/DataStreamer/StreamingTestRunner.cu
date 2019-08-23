#include <common/cuda/DisableBoostOnDevice.h>

#include "StreamingTestRunner.h"

#include <dataTypes/BatchData.cuh>
#include <common/cuda/streams/LaunchManager.cuh>
#include <common/DataGenerators/TemplateGenerator.h>

#include <cuda_runtime.h>

namespace PacBio {
namespace Cuda {

using namespace PacBio::Cuda::Data;

// Just provides basic validation of receipt.  Iterates through all the data to
// make sure it has the expected value as per TemplateGenerator.  Can be easily tweaked
// to loop over the data multiple times, to artificially inflate compute time.
// function implicitly assumes that the kernel blocks only contain 32 threads.
__global__ void BasicSanity(Mongo::Data::GpuBatchData<PBShort2> in, size_t tid, Memory::DeviceView<size_t> ret)
{
    const size_t reps = 10;
    auto zmwData = in.ZmwData(blockIdx.x, threadIdx.x);
    for (size_t i = 0; i < reps; ++i)
    {
        short val = zmwData[0].X();
        bool valid = true;
        for (auto& data : zmwData)
        {
            auto myValid = (data.X() == val) && (data.Y() == val);
            auto warpValid = __all_sync(0xFFFF, myValid);
            if (threadIdx.x == 0)
                valid = valid && warpValid;
        }
        if (threadIdx.x == 0)
           ret[blockIdx.x] = valid ? 1 : 0;
    }
}

struct ThreadRunner
{
    template <typename Func, typename... Args>
    ThreadRunner(Func&& f, Args... args)
    {
        tid = ++instanceCount_;
        t = std::thread(std::forward<Func>(f), tid, std::forward<Args>(args)...);
    }

    ThreadRunner(const ThreadRunner&) = delete;
    ThreadRunner(ThreadRunner&&) = default;
    ThreadRunner& operator=(const ThreadRunner&) = delete;
    ThreadRunner& operator=(ThreadRunner&&) = default;

    ~ThreadRunner()
    {
        if (t.joinable()) t.join();
    }
private:
    std::thread t;
    size_t tid;
    static std::atomic<size_t> instanceCount_;
};

std::atomic<size_t> ThreadRunner::instanceCount_{0};

void RunTest(const Data::DataManagerParams& params, size_t simulKernels)
{
    ZmwDataManager<short> manager(params, std::make_unique<TemplateGenerator>(params));

    auto func = [&manager, &params](size_t tid) {
        Memory::UnifiedCudaArray<size_t> ret(params.kernelLanes, Memory::SyncDirection::HostReadDeviceWrite);
        while (manager.MoreData())
        {
            try {
                auto batch = manager.NextBatch();
                const auto& launcher = PBLauncher(BasicSanity,
                                                params.kernelLanes,
                                                params.laneWidth/2);
                launcher(batch.KernelInput(), tid, ret);

                auto view = ret.GetHostView();
                bool valid = true;
                for (size_t i = 0; i < params.kernelLanes; ++i)
                {
                    valid = valid && view[i];
                    view[i] = 0;
                }
                if (!valid)
                    std::cerr << "Thread " << tid << " failed validation\n";
                manager.ReturnBatch(std::move(batch));
            } catch (std::runtime_error& e) {
                // Currently expected that we might throw if we learn there is no more data after
                // we've already tried to fetch the next batch.  That is benign, any other exception
                // implies a problem
                if (manager.MoreData())
                {
                    std::cerr << e.what() << std::endl;
                    throw PBException("Unexpected exception processing data");
                }
            }
        }
    };

    // Set up a threadpool to have them each asynchronously launch cuda kernels
    // to process the input data stream.  The destructor calls at the end of the
    // function will automatically wait and join the threads.
    std::vector<ThreadRunner> threads;
    threads.reserve(simulKernels);
    for(size_t i = 0; i < simulKernels; ++i)
    {
        threads.emplace_back(func);
    }
}

}}
