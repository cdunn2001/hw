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

#include <common/cuda/streams/CudaEvent.h>

using namespace PacBio::Cuda;

namespace {
// Dummy kernel to just burn time
__global__ void SpinKernel(uint64_t duration)
{
    auto start = clock64();
    while (clock64() - start < duration) {}
    return;
}

}

TEST(CudaEvent, NoKernel)
{
    CudaEvent event;
    EXPECT_TRUE(event.IsCompleted());
}

TEST(CudaEvent, ManualSync)
{
    CudaEvent event;
    EXPECT_TRUE(event.IsCompleted());

    SpinKernel<<<1,1>>>(1000000);
    event.RecordEvent();
    EXPECT_FALSE(event.IsCompleted());

    CudaSynchronizeDefaultStream();
    EXPECT_TRUE(event.IsCompleted());
}

TEST(CudaEvent, AutoSync)
{
    CudaEvent event;
    EXPECT_TRUE(event.IsCompleted());

    SpinKernel<<<1,1>>>(1000000);
    event.RecordEvent();
    EXPECT_FALSE(event.IsCompleted());

    event.WaitForCompletion();
    EXPECT_TRUE(event.IsCompleted());
}

TEST(CudaEvent, MultiStream)
{
    CudaEvent event1;
    CudaEvent event2;

    // set up two threads that execute two
    // concurrent kernels
    {
        std::thread t2([&](){
                SpinKernel<<<1,1>>>(10000000);
                event2.RecordEvent();
            });

        std::thread t1([&](){
                SpinKernel<<<1,1>>>(5000000);
                event1.RecordEvent();
            });

        t1.join();
        t2.join();
    }

    EXPECT_FALSE(event1.IsCompleted());
    EXPECT_FALSE(event2.IsCompleted());

    event1.WaitForCompletion();
    EXPECT_TRUE(event1.IsCompleted());
    EXPECT_FALSE(event2.IsCompleted());

    event2.WaitForCompletion();
    EXPECT_TRUE(event1.IsCompleted());
    EXPECT_TRUE(event2.IsCompleted());
}
