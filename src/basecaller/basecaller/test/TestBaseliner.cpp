
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
//  Defines unit tests for the strategies for estimation and subtraction of
//  baseline and estimation of associated statistics.

#include <basecaller/traceAnalysis/BaselineEstimators.h>
#include <common/DataGenerators/BatchGenerator.h>

#include <gtest/gtest.h>

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TEST(TestNoOpBaseliner, Run)
{
    NoOpBaseliner foo{0};

    Cuda::Data::BatchGenerator batchGenerator(128, laneSize, 1, 8192, 1);

    while (!batchGenerator.Finished())
    {
        auto chunk = batchGenerator.PopulateChunk();
        Data::CameraTraceBatch cameraBatch = foo(std::move(chunk.front()));
        auto baselineStats = cameraBatch.Stats(0).BaselineStats();
        EXPECT_TRUE(all(NoOpBaseliner::FloatArray{0} == baselineStats.Count()));
        EXPECT_TRUE(all(isnan(baselineStats.Mean())));
        EXPECT_TRUE(all(isnan(baselineStats.Variance())));
    }

}

}}} // PacBio::Mongo::Basecaller
