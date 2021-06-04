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

#include <gtest/gtest.h>

#include <common/cuda/utility/CudaArray.h>

using namespace PacBio::Cuda::Utility;

TEST(CudaArray, Values)
{
    CudaArray<int, 3> arr{10};

    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 10);
    EXPECT_EQ(arr[2], 10);

    arr[1] = 5;
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 5);
    EXPECT_EQ(arr[2], 10);

    arr[2] = 15;
    EXPECT_EQ(arr[0], 10);
    EXPECT_EQ(arr[1], 5);
    EXPECT_EQ(arr[2], 15);
}

TEST(CudaArray, Iterators)
{
    CudaArray<int, 12> arr{1};

    EXPECT_EQ(arr.size(), 12);
    EXPECT_EQ(arr.begin(), arr.data());
    EXPECT_EQ(arr.end(), arr.begin() + arr.size());

    EXPECT_EQ(&arr[3],arr.begin() + 3);

    int sum = 0;
    for (const auto& v : arr) sum += v;
    EXPECT_EQ(sum, 12);
}

TEST(CudaArray, Construct)
{
    std::array<int, 4> arr1 {1,2,3,4};

    CudaArray<int, 4> arr2(arr1);
    EXPECT_EQ(arr2[0], 1);
    EXPECT_EQ(arr2[1], 2);
    EXPECT_EQ(arr2[2], 3);
    EXPECT_EQ(arr2[3], 4);

    CudaArray<int, 3> arr3{3};
    EXPECT_EQ(arr3[0], 3);
    EXPECT_EQ(arr3[1], 3);
    EXPECT_EQ(arr3[2], 3);
}
