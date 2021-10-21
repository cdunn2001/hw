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
//

#include <pacbio/datasource/MallocAllocator.h>
#include <common/BatchDataSource.h>

#include <gtest/gtest.h>

using namespace PacBio::Mongo;
using namespace PacBio::DataSource;

namespace {

class TestDataSource : public BatchDataSource
{
    static constexpr int lanesPerPacket = 4;
    static constexpr int framesPerChunk = 64;
    static DataSourceBase::Configuration CreateConfig()
    {
        PacketLayout layout(PacketLayout::LayoutType::BLOCK_LAYOUT_DENSE,
                            PacketLayout::EncodingFormat::INT16,
                            {lanesPerPacket, framesPerChunk, laneSize});
        return DataSourceBase::Configuration(layout, std::make_unique<MallocAllocator>());
    }
public:
    TestDataSource(uint32_t poolsPerChip, uint32_t chunksPerMovie)
        : BatchDataSource(CreateConfig())
        , poolsPerChip_(poolsPerChip)
        , chunksPerMovie_(chunksPerMovie)
        , currChunk_(0, framesPerChunk)
    {}

    std::map<uint32_t, PacketLayout> PacketLayouts() const override { throw PBException("Not Implemented"); };
    std::vector<UnitCellProperties> GetUnitCellProperties() const override { throw PBException("Not Implemented"); }
    std::vector<uint32_t> UnitCellIds() const override { throw PBException("Not Implemented"); }
    size_t NumFrames() const override { throw PBException("Not Implemented"); }
    size_t NumZmw() const override { throw PBException("Not Implemented"); }
    double FrameRate() const override { throw PBException("Not Implemented"); }
    HardwareInformation GetHardwareInformation() override { throw PBException("Not Implemented"); }
    int16_t Pedestal() const override { throw PBException("Not Implemented"); }

private:
    void ContinueProcessing() override
    {
        const auto& layout = GetConfig().requestedLayout;
        currChunk_.AddPacket(SensorPacket{layout,
                currChunk_.NumPackets(),
                currChunk_.NumPackets() * layout.NumZmw(),
                currFrame_,
                *GetConfig().allocator});

        if (currChunk_.NumPackets() == poolsPerChip_)
        {
            currFrame_ += framesPerChunk;
            PushChunk(std::move(currChunk_));
            currChunk_ = SensorPacketsChunk(currFrame_, currFrame_ + framesPerChunk);
            ++chunkCount_;
            if (chunkCount_ == chunksPerMovie_)
            {
                this->SetDone();
            }
        }
    };

    uint32_t poolsPerChip_;
    uint32_t chunksPerMovie_;
    uint32_t chunkCount_ = 0;
    uint32_t currFrame_ = 0;
    SensorPacketsChunk currChunk_;
};

}

TEST(TestDataSource, LoopOverChunks)
{
    const uint32_t poolsPerChip = 10;
    const uint32_t chunksPerMovie = 5;
    TestDataSource source{poolsPerChip, chunksPerMovie};

    uint32_t seenChunks = 0;
    for (const auto& chunk : source.AllChunks())
    {
        EXPECT_EQ(chunk.size(), poolsPerChip);
        ++seenChunks;
    }
    EXPECT_EQ(seenChunks, chunksPerMovie);
};

TEST(TestDataSource, LoopOverBatches)
{
    const uint32_t poolsPerChip = 10;
    const uint32_t chunksPerMovie = 5;
    TestDataSource source{poolsPerChip, chunksPerMovie};

    uint32_t seenChunks = 0;
    uint32_t seenBatches = 0;
    for (const auto& batch : source.AllBatches())
    {
        EXPECT_EQ(batch.Metadata().PoolId(), seenBatches);
        ++seenBatches;
        if (seenBatches == poolsPerChip)
        {
            seenBatches = 0;
            ++seenChunks;
        }
    }
    EXPECT_EQ(seenChunks, chunksPerMovie);
};
