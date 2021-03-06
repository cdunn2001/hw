// Copyright (c) 2020-2021 Pacific Biosciences of California, Inc.
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

#include <algorithm>
#include <random>

#include <gtest/gtest.h>

#include <pacbio/datasource/SensorPacket.h>
#include <pacbio/datasource/SensorPacketsChunk.h>
#include <pacbio/datasource/MallocAllocator.h>
#include <pacbio/utilities/SmartEnum.h>

#include <common/graphs/GraphManager.h>
#include <common/graphs/GraphNodeBody.h>
#include <common/graphs/GraphNode.h>

#include <appModules/BlockRepacker.h>
#include <dataTypes/TraceBatch.h>

using namespace PacBio::Application;
using namespace PacBio::Graphs;
using namespace PacBio::Logging;
using namespace PacBio::Memory;
using namespace PacBio::DataSource;
using namespace PacBio::Mongo::Data;

namespace {

// Describes the size and frequency of "irregular" packets.
// Irregular packets have the same block layout, but
// fewer (or I guess extra?) blocks.  I'm not even sure this
// functionality is necessary, but I'm testing it anyway in
// case the datasource is incapable of having a sensor packet
// wrap around the edge of a chip.
struct IrregularPacketInfo
{
    size_t frequency = 0;
    size_t numBlocks = 0;
};

template <typename T>
SensorPacket GeneratePacket(const PacketLayout& layout,
                            size_t packetID,
                            size_t startZmw,
                            size_t startFrame,
                            IAllocator& alloc)
{
    SensorPacket packet(layout, packetID, startZmw, startFrame, alloc);
    packetID++;

    static constexpr auto numBits = sizeof(T) * 8 / 2;
    static constexpr auto moduloMask = (1<<numBits)-1;

    //populate data
    for (size_t b = 0; b < layout.NumBlocks(); ++b)
    {
        auto* bdata = reinterpret_cast<T*>(packet.BlockData(b).Data());
        for (size_t frame = 0; frame < layout.NumFrames(); ++frame)
        {
            for (size_t zmw = 0; zmw < layout.BlockWidth(); ++zmw)
            {
                auto zdata = (startZmw + zmw + b * layout.BlockWidth()) & moduloMask;
                auto fdata = (startFrame + frame) & moduloMask;
                bdata[zmw + frame*layout.BlockWidth()] = (fdata << numBits) | zdata;
            }
        }
    }
    return packet;
}

// Populates a chunk full of sensor packets.  Most packets will
// have the shape of `defaultLayout`, but `irregular` can be
// used to sneak in periodic chunks with more or fewer blocks.
//
// Will use a very basic test pattern where the low half
// holds the (modulo) zmw number, and the high half holds the
// (modulo) frame index.
SensorPacketsChunk GeneratePacketsChunk(PacketLayout defaultLayout,
                                        IrregularPacketInfo irregular,
                                        size_t startFrame,
                                        size_t numFrames,
                                        size_t numZmw)
{
    const auto zmwPerCacheLine = [&](){
        switch (defaultLayout.Encoding())
        {
        case PacBio::DataSource::PacketLayout::INT16:
            return 32;
        case PacBio::DataSource::PacketLayout::UINT8:
            return 64;
        default:
            throw PBException("Misconfigured test");
        }
    }();
    if (numZmw % zmwPerCacheLine != 0) throw PBException("numZmw must be a multiple of a cache line");
    if (defaultLayout.BlockWidth() % zmwPerCacheLine != 0) throw PBException("block width must be a multiple of a cache line");
    if (startFrame % defaultLayout.NumFrames()) throw PBException("startFrame must be evenly divisible by the layout frames");
    if (numFrames % defaultLayout.NumFrames() != 0) throw PBException("numFrames must be evenly divisible by the layout frames");
    if (irregular.frequency != 0 && irregular.numBlocks == 0) throw PBException("Invalid specification for irregular");

    MallocAllocator alloc;

    SensorPacketsChunk ret(startFrame, startFrame + numFrames);
    ret.SetZmwRange(0, numZmw);

    const size_t numTimePackets = numFrames / defaultLayout.NumFrames();
    size_t packetID = 0;
    for (size_t i = 0; i < numTimePackets; ++i)
    {
        size_t currZmw = 0;
        size_t currPacket = 1;
        while (currZmw < numZmw)
        {
            size_t numBlocks;
            if (irregular.frequency != 0 && currPacket % irregular.frequency)
            {
                numBlocks = irregular.numBlocks;
            } else
            {
                numBlocks = defaultLayout.NumBlocks();
            }
            const size_t endZmw = std::min(numZmw, currZmw + numBlocks * defaultLayout.BlockWidth());
            numBlocks = (endZmw - currZmw) / defaultLayout.BlockWidth();

            PacketLayout layout(PacketLayout::BLOCK_LAYOUT_DENSE, defaultLayout.Encoding(),
                                {numBlocks, defaultLayout.NumFrames(), defaultLayout.BlockWidth()});

            SensorPacket packet;
            switch (defaultLayout.Encoding())
            {
            case PacBio::DataSource::PacketLayout::INT16:
                packet = GeneratePacket<int16_t>(layout,
                                                 packetID,
                                                 currZmw,
                                                 startFrame + i*defaultLayout.NumFrames(),
                                                 alloc);
                break;
            case PacBio::DataSource::PacketLayout::UINT8:
                packet = GeneratePacket<uint8_t>(layout,
                                                 packetID,
                                                 currZmw,
                                                 startFrame + i*defaultLayout.NumFrames(),
                                                 alloc);
                break;
            default:
                throw PBException("Unexpected layout in TestBlockRepacker");
            }
            packetID++;

            ret.AddPacket(std::move(packet));
            currZmw = endZmw;
        }
    }

    // Re-order the packets, just for the fun of it. Don't want
    // the order to change between runs, nor do we really care
    // if the shuffle is all that random.  We just want to
    // present data that aren't already sorted
    std::mt19937 g(12345);
    std::shuffle(ret.begin(), ret.end(), g);

    return ret;
}

// Generator where the irregularities are added automatically.
// I was originally going to have explicit irregularity settings
// in the parameter sweeps, but the unit tests were taking too
// long to run even before then.  So now all tests get a small
// ammount implicitly.
SensorPacketsChunk GeneratePacketsChunk(PacketLayout defaultLayout,
                                        size_t startFrame,
                                        size_t numFrames,
                                        size_t numZmw)
{
    IrregularPacketInfo irregular;
    irregular.frequency = 5;
    irregular.numBlocks = defaultLayout.NumBlocks()-3;
    return GeneratePacketsChunk(defaultLayout,
                                irregular,
                                startFrame,
                                numFrames, numZmw);
}

// Keeps track of information as we stream through a chunk,
// making sure that:
// 1. All incoming batches are of the expected size
// 2. The entire chunk is actually seen
// 3. The data has the expected pattern
class Validator
{
public:
    Validator(BatchDimensions dims, size_t numZmw)
        : dims_(dims)
    {
        auto numBlocks = numZmw / dims.laneWidth;
        validBlocks_ = std::vector<bool>(numBlocks, false);
        seenBlocks_ = std::vector<bool>(numBlocks, false);
    }

    template <typename T>
    void Validate(const TraceBatch<T>& in)
    {
        if (in.StorageDims().framesPerBatch != dims_.framesPerBatch) batchError_ = true;
        if (in.StorageDims().laneWidth != dims_.laneWidth) batchError_ = true;
        if (in.StorageDims().lanesPerBatch != dims_.lanesPerBatch) batchError_ = true;
        if (in.Metadata().FirstFrame() % dims_.framesPerBatch != 0) batchError_ = true;
        if (in.Metadata().FirstZmw() != in.Metadata().PoolId() * dims_.ZmwsPerBatch()) batchError_ = true;

        if (batchError_) return;
        if(batchError_) assert(false);

        static constexpr auto numBits = sizeof(T) * 8 / 2;
        static constexpr auto moduloMask = (1<<numBits)-1;

        const size_t bIdx = in.Metadata().FirstZmw() / dims_.laneWidth;
        for (size_t b = 0; b < in.LanesPerBatch(); ++b)
        {
            bool blockError = false;
            if (seenBlocks_[bIdx + b]) blockError = true;
            seenBlocks_[bIdx + b] = true;

            assert(!blockError);

            auto block = in.GetBlockView(b);
            for (size_t frame = 0; frame < dims_.framesPerBatch; ++frame)
            {
                for (size_t zmw = 0; zmw < dims_.laneWidth; ++zmw)
                {
                    auto zdata = (in.Metadata().FirstZmw() + zmw + b * dims_.laneWidth) & moduloMask;
                    auto fdata = (frame + in.GetMeta().FirstFrame()) & moduloMask;
                    T answer = static_cast<T>((fdata << numBits) | zdata);
                    T result = block[zmw + frame*dims_.laneWidth];
                    if (result != answer) blockError = true;
                    assert(!blockError);
                }
            }
            validBlocks_[bIdx + b] = !blockError;
        }
    }

    bool AllCorrect()
    {
        if (batchError_)
        {
            PBLOG_ERROR << "Batch of unexpected shape/location was seen";
            return false;
        }

        bool allSeen = true;
        for (const auto& seen : seenBlocks_) allSeen &= seen;
        if (!allSeen)
        {
            PBLOG_ERROR << "Not all blocks in the chunk were seen";
            return false;
        }

        size_t invalidCount = 0;
        for (const auto& val : validBlocks_)
        {
            if (!val) invalidCount++;
        }
        if (invalidCount > 0)
        {
            PBLOG_ERROR << invalidCount << " blocks were incorrect\n";
        }
        return invalidCount == 0;
    }

    void Reset()
    {
        batchError_ = false;
        for (size_t i = 0; i < validBlocks_.size(); ++i)
        {
            validBlocks_[i] = false;
            seenBlocks_[i] = false;
        }
    }
private:
    bool batchError_ = false;
    BatchDimensions dims_;
    std::vector<bool> validBlocks_;
    std::vector<bool> seenBlocks_;
};

// Small class, to plug the above validator into a compute graph
class ValidatorBody final : public LeafBody<const TraceBatchVariant>
{
public:
    ValidatorBody(Validator* validator)
        : validator_(validator)
    {}

    size_t ConcurrencyLimit() const override { return 1; }
    float MaxDutyCycle() const override { return 1.0f; }

    void Process(const TraceBatchVariant& in) override
    {
        std::visit([&](const auto& batch) { validator_->Validate(batch);},
                   in.Data());
    }
private:
    Validator* validator_;
};

// Set up some "strong typedefs" so we can
// do better than parameterizing our test on
// a whole bunch of `size_t`
template <typename T>
struct Wrapper
{
    explicit Wrapper(T t)
    : data_(t)
    {}
    operator const T&() const { return data_;}
private:
    T data_;
};

struct NumZmw : public Wrapper<size_t> { using Wrapper::Wrapper; };
struct NumFrames : public Wrapper<size_t> { using Wrapper::Wrapper; };
struct BatchBlocks : public Wrapper<size_t> { using Wrapper::Wrapper; };
struct PacketFrames : public Wrapper<size_t> { using Wrapper::Wrapper; };
struct PacketBlocks : public Wrapper<size_t> { using Wrapper::Wrapper; };
struct PacketBlockWidth : public Wrapper<size_t> { using Wrapper::Wrapper; };
struct Encoding : public Wrapper<PacketLayout::EncodingFormat> { using Wrapper::Wrapper; };

}

struct TestBlockRepacker : testing::TestWithParam<std::tuple<NumZmw, NumFrames, BatchBlocks,
                                                             PacketFrames, PacketBlocks,
                                                             PacketBlockWidth, Encoding
                                                             >
                                                  >
{};

TEST_P(TestBlockRepacker, ThreeChunkRepack)
{
    LogSeverityContext context(LogLevel::WARN);

    const size_t   numZmw = std::get<NumZmw>(GetParam());
    const uint32_t numFrames = std::get<NumFrames>(GetParam());
    const uint32_t batchBlocks = std::get<BatchBlocks>(GetParam());
    const size_t   blockWidth = 64;

    const uint32_t packetFrames = std::get<PacketFrames>(GetParam());
    const size_t   packetBlocks = std::get<PacketBlocks>(GetParam());
    const size_t   packetBlockWidth = std::get<PacketBlockWidth>(GetParam());

    const auto     encoding = std::get<Encoding>(GetParam());

    const size_t numThreads = 4;

    PacketLayout inputLayout(PacketLayout::BLOCK_LAYOUT_DENSE, encoding, {packetBlocks, packetFrames, packetBlockWidth});
    BatchDimensions outputDims {batchBlocks, numFrames, blockWidth};
    auto chunk1 = GeneratePacketsChunk(inputLayout, 0*numFrames, numFrames, numZmw);
    auto chunk2 = GeneratePacketsChunk(inputLayout, 1*numFrames, numFrames, numZmw);
    auto chunk3 = GeneratePacketsChunk(inputLayout, 2*numFrames, numFrames, numZmw);

    ASSERT_TRUE(chunk1.IsValid());
    ASSERT_TRUE(chunk2.IsValid());
    ASSERT_TRUE(chunk3.IsValid());

    SMART_ENUM(GraphProfiler, REPACKER, VALIDATOR);

    Validator validator(outputDims, numZmw);

    // Set up a toy graph, which just repacks a packet
    // and puts the resulting batch through some
    // validation code
    std::map<uint32_t, PacketLayout> layoutMap;
    for (const auto& p : chunk1)
    {
        layoutMap[p.PacketID()] = p.Layout();
    }
    GraphManager <GraphProfiler> graph(numThreads);
    auto* repacker = graph.AddNode(std::make_unique<BlockRepacker>(layoutMap, outputDims, numZmw, numThreads), GraphProfiler::REPACKER);
    repacker->AddNode(std::make_unique<ValidatorBody>(&validator), GraphProfiler::VALIDATOR);

    for (auto& packet : chunk1)
    {
        repacker->ProcessInput(std::move(packet));
    }
    graph.Synchronize();
    EXPECT_TRUE(validator.AllCorrect());
    validator.Reset();

    for (auto& packet : chunk2)
    {
        repacker->ProcessInput(std::move(packet));
    }
    graph.Synchronize();
    EXPECT_TRUE(validator.AllCorrect());
    validator.Reset();

    for (auto& packet : chunk3)
    {
        repacker->ProcessInput(std::move(packet));
    }
    graph.Synchronize();
    EXPECT_TRUE(validator.AllCorrect());
}

static std::string TestNameGenerator(const testing::TestParamInfo<TestBlockRepacker::ParamType>& info)
{
    std::string ret;
    ret += "Zmw" + std::to_string(std::get<NumZmw>(info.param)) + "_";
    ret += "Frames" + std::to_string(std::get<NumFrames>(info.param)) + "_";
    ret += "BatchBlocks" + std::to_string(std::get<BatchBlocks>(info.param)) + "_";
    ret += "PacketFrames" + std::to_string(std::get<PacketFrames>(info.param)) + "_";
    ret += "PacketBlocks" + std::to_string(std::get<PacketBlocks>(info.param)) + "_";
    ret += "PacketBlockWidth" + std::to_string(std::get<PacketBlockWidth>(info.param));
    return ret;
};

INSTANTIATE_TEST_SUITE_P(INT16ParamSweep,
    TestBlockRepacker,
    ::testing::Combine(
        ::testing::Values(16384, 163840),// NumZmw
        ::testing::Values(256),          // NumFrames
        ::testing::Values(16, 256),      // BatchBlocks
        ::testing::Values(64, 256),      // PacketFrames
        ::testing::Values(50),           // PacketBlocks
        ::testing::Values(32, 64, 128),  // PacketBlockWidth
        ::testing::Values(PacketLayout::INT16)),
    TestNameGenerator);

INSTANTIATE_TEST_SUITE_P(UINT8ParamSweep,
    TestBlockRepacker,
    ::testing::Combine(
        ::testing::Values(16384, 163840),// NumZmw
        ::testing::Values(256),          // NumFrames
        ::testing::Values(16, 256),      // BatchBlocks
        ::testing::Values(64, 256),      // PacketFrames
        ::testing::Values(50),           // PacketBlocks
        // 8 bit repacking can't do 32 pixel lanes
        ::testing::Values(64, 128),      // PacketBlockWidth
        ::testing::Values(PacketLayout::UINT8)),
    TestNameGenerator);
