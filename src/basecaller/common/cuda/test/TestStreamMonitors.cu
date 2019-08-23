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

#include <thread>

#include <gtest/gtest.h>

#include <common/cuda/streams/KernelLaunchInfo.h>
#include <common/cuda/streams/LaunchManager.cuh>

#include <common/cuda/memory/UnifiedCudaArray.h>
#include <common/cuda/memory/DeviceOnlyArray.cuh>

using namespace PacBio::Cuda;
using namespace PacBio::Cuda::Memory;
using namespace PacBio::Logging;

namespace {
// Dummy kernel to just burn time, so we can have controlled
// exploration of multi-stream issues
template <typename T>
__global__ void SpinKernel(uint64_t duration, DeviceView<T>)
{
    auto start = clock64();
    while (clock64() - start < duration) {}
    return;
}

}

// Note: These tests are testing the behaviour of UnifiedCudaArray
//       and DeviceOnlyArray.  Preferrably they'd be testing the
//       SingleStreamMonitor and MultiStreamMonitor implementations
//       directly, but that is currently difficult.  They rely on
//       consuming KernelLaunchInfo objects, which are quite
//       intentionally impossible for anyone outside the launcher
//       framework to construct.  Hopefully I can come up with a good
//       testing hook to get at these classes more directly without
//       compromising their integrety, but for now this tests their
//       functionality robustly, if by proxy.

// The SingleStreamMonitor can be moved between streams, as long
// as it's not simultaneously used by two kernels.
TEST(StreamMonitor, UnifiedCudaArray_Good)
{
    PBLogger::SetMinimumSeverityLevel(LogLevel::ULTIMATE);
    ResetStreamErrors();
    auto arr = std::make_unique<UnifiedCudaArray<int>>(1, SyncDirection::HostWriteDeviceRead);

    std::thread t1([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
            CudaSynchronizeDefaultStream();
        });
    t1.join();

    std::thread t2([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
            CudaSynchronizeDefaultStream();
        });
    t2.join();

    arr.reset();
    EXPECT_EQ(StreamErrorCount(), 0);
}

// Simultaneous use in two streams is an error
TEST(StreamMonitor, UnifiedCudaArray_ConcurrentAccess)
{
    PBLogger::SetMinimumSeverityLevel(LogLevel::ULTIMATE);
    ResetStreamErrors();
    auto arr = std::make_unique<UnifiedCudaArray<int>>(1, SyncDirection::HostWriteDeviceRead);

    std::thread t1([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
        });
    t1.join();

    std::thread t2([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000, *arr);
            EXPECT_EQ(StreamErrorCount(), 1);
        });
    t2.join();
}

// Also is an error to delete the object while used in
// a live kernel
TEST(StreamMonitor, UnifiedCudaArray_ConcurrentDelete)
{
    PBLogger::SetMinimumSeverityLevel(LogLevel::ULTIMATE);
    ResetStreamErrors();
    auto arr = std::make_unique<UnifiedCudaArray<int>>(1, SyncDirection::HostWriteDeviceRead);

    std::thread t1([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
        });
    t1.join();

    arr.reset();
    EXPECT_EQ(StreamErrorCount(), 1);
}

// A DeviceOnlyArray of *non* const data still cannot
// be shared between different kernels
TEST(StreamMonitor, DeviceOnlyArray_ConcurrentAccess)
{
    PBLogger::SetMinimumSeverityLevel(LogLevel::ULTIMATE);
    ResetStreamErrors();
    auto arr = std::make_unique<DeviceOnlyArray<int>>(1, 1);

    std::thread t1([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000000ull, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
        });
    t1.join();

    std::thread t2([&](){
            PBLauncher(SpinKernel<int>, 1, 1)(1000000000ull, *arr);
            EXPECT_EQ(StreamErrorCount(), 1);
        });
    t2.join();
}

// A *const* DeviceOnlyArray *can* be shared between concurrent
// kernels
TEST(StreamMonitor, DeviceOnlyArray_ConstConcurrentAccess)
{
    PBLogger::SetMinimumSeverityLevel(LogLevel::ULTIMATE);
    ResetStreamErrors();
    auto arr = std::make_unique<DeviceOnlyArray<const int>>(1, 1);

    std::thread t1([&](){
            PBLauncher(SpinKernel<const int>, 1, 1)(1000000000ull, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
        });
    t1.join();

    std::thread t2([&](){
            PBLauncher(SpinKernel<const int>, 1, 1)(1000000000ull, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
        });
    t2.join();
}

// Again make sure that we cannot delete the object while
// a kernel is executing.
TEST(StreamMonitor, DeviceOnlyArray_ConcurrentDelete)
{
    PBLogger::SetMinimumSeverityLevel(LogLevel::ULTIMATE);
    ResetStreamErrors();
    auto arr = std::make_unique<DeviceOnlyArray<const int>>(1, 1);

    std::thread t1([&](){
            PBLauncher(SpinKernel<const int>, 1, 1)(1000000000ull, *arr);
            EXPECT_EQ(StreamErrorCount(), 0);
        });
    t1.join();

    arr.reset();
    EXPECT_EQ(StreamErrorCount(), 1);
}
