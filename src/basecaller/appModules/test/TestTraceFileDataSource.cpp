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

#include <appModules/TraceFileDataSource.h>
#include <pacbio/datasource/MallocAllocator.h>

using namespace PacBio::Application;
using namespace PacBio::DataSource;
using namespace PacBio::Memory;
using namespace PacBio::Mongo;

struct TestTraceFileDataSource : testing::TestWithParam<std::pair<bool, std::string>> {};

TEST_P(TestTraceFileDataSource, Read8And16)
{
    PacBio::Logging::LogSeverityContext severity(PacBio::Logging::LogLevel::ERROR);

    const uint32_t lanesPerPool = 64;
    const uint32_t framesPerBlock = 512;

    PacketLayout layout16(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::INT16,
                        {lanesPerPool, framesPerBlock, laneSize});
    PacketLayout layout8(PacketLayout::BLOCK_LAYOUT_DENSE, PacketLayout::UINT8,
                        {lanesPerPool, framesPerBlock, laneSize});

    DataSourceBase::Configuration cfg16(layout16, std::make_unique<MallocAllocator>());
    DataSourceBase::Configuration cfg8(layout8, std::make_unique<MallocAllocator>());

    const std::string trc = GetParam().second;
    TraceFileDataSource source16(std::move(cfg16),
                                 trc,
                                 framesPerBlock,
                                 lanesPerPool*laneSize,
                                 GetParam().first);
    TraceFileDataSource source8(std::move(cfg8),
                                trc,
                                framesPerBlock,
                                lanesPerPool*laneSize,
                                GetParam().first);

    while (!source16.ChunksReady()) source16.ContinueProcessing();
    while (!source8.ChunksReady()) source8.ContinueProcessing();

    SensorPacketsChunk chunk16;
    SensorPacketsChunk chunk8;

    source16.PopChunk(chunk16, std::chrono::milliseconds{100});
    source8.PopChunk(chunk8, std::chrono::milliseconds{100});

    ASSERT_EQ(chunk16.NumPackets(), 1);
    ASSERT_EQ(chunk8.NumPackets(), 1);

    SensorPacket& packet16 = *chunk16.begin();
    SensorPacket& packet8 = *chunk8.begin();

    ASSERT_EQ(packet8.Layout().BlockWidth(), packet16.Layout().BlockWidth());
    ASSERT_EQ(packet8.Layout().NumBlocks(), packet16.Layout().NumBlocks());
    ASSERT_EQ(packet8.Layout().NumFrames(), packet16.Layout().NumFrames());
    ASSERT_EQ(packet8.Layout().Type(), PacketLayout::BLOCK_LAYOUT_DENSE);
    ASSERT_EQ(packet16.Layout().Type(), PacketLayout::BLOCK_LAYOUT_DENSE);
    ASSERT_EQ(packet8.Layout().Encoding(), PacketLayout::UINT8);
    ASSERT_EQ(packet16.Layout().Encoding(), PacketLayout::INT16);

    ASSERT_EQ(packet16.BytesInBlock(), 2*packet8.BytesInBlock());

    size_t skipped = 0;
    size_t checked = 0;
    for (uint32_t i = 0; i < packet16.Layout().NumBlocks(); ++i)
    {
        uint8_t* data8 = packet8.BlockData(i).Data();
        int16_t* data16 = reinterpret_cast<int16_t*>(packet16.BlockData(i).Data());

        for (uint32_t j = 0; j < packet8.BytesInBlock(); ++j)
        {
            if (data16[j] < 0)
            {
                EXPECT_EQ(data8[j], 0);
                skipped++;
            } else if (data16[j] > 255)
            {
                EXPECT_EQ(data8[j], 255);
                skipped++;
            } else {
                EXPECT_EQ(data8[j], static_cast<int8_t>(data16[j]))
                    << j << " " << data8[j] << " " << data16[j];
                checked++;
            }
        }
    }
    // Just a quick check to make sure the data file was suitable,
    // and we aren't truncating/saturating more than not.
    EXPECT_GT(checked, skipped);
}

INSTANTIATE_TEST_SUITE_P(,
                         TestTraceFileDataSource,
                         ::testing::Values(std::make_pair(true,std::string{"/pbi/dept/primary/unitTestInput/mongo/test.trc.h5"}),
                                           std::make_pair(false,std::string{"/pbi/dept/primary/unitTestInput/mongo/test.trc.h5"})),
                                           [](const testing::TestParamInfo<std::pair<bool, std::string>>& info) {
                             std::string ret;
                             if (info.param.first) ret = "Cached";
                             else ret = "NotCached";
                             if (info.param.second.find("test.trc.h5") != std::string::npos) ret += "16BitInput";
                             else ret += "8BitInput";
                             return ret;
                         });
