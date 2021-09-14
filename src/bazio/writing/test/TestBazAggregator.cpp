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

#include <bazio/writing/BazAggregator.h>
#include <appModules/Metrics.h>

using namespace PacBio::BazIO;
using namespace PacBio::Application;

using BazAggregatorT = BazAggregator<ProductionMetricsGroup::MetricT,
                                     ProductionMetricsGroup::MetricAggregatedT>;

// BazAggregator is currently just a thin wrapper around
// PacketBufferManager, and may go away.  For now, I'm
// just testing the only new feature it provides, which
// is the ability it iterate over nonHQ zmw (to potentially)
// decide to mark them as HQ
TEST(BazAggregator, Iterators)
{
    uint32_t numZmw = 12;
    uint32_t bytesPerZmw = 10;
    BazAggregatorT agg(numZmw, 0, bytesPerZmw, 1, true);

    EXPECT_EQ(agg.NumZmw(), numZmw);
    EXPECT_EQ(agg.PreHQData().size(), numZmw);

    int i = 0;
    for (auto&& zmw : agg.PreHQData())
    {
        if (i % 3 == 0) zmw.MarkAsHQ();
        i++;
    }

    EXPECT_EQ(agg.PreHQData().size(), 2*numZmw/3);
    for (auto&& zmw : agg.PreHQData())
    {
        zmw.MarkAsHQ();
    }

    EXPECT_EQ(agg.PreHQData().size(), 0);
}
