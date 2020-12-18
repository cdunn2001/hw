//
// Created by mlakata on 12/3/20.
//

#include <vector>
#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/logging/Logger.h>
#include <pacbio/dev/TemporaryDirectory.h>

#include <appModules/Basecaller.h>
#include <dataTypes/configs/MovieConfig.h>
#include <dataTypes/TraceBatch.h>
#include <pacbio/sensor/ChipLayoutSimple.h>
#include <appModules/TraceSaver.h>
#include <pacbio/tracefile/TraceFile.h>
#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/datasource/MallocAllocator.h>

using std::vector;

using namespace PacBio;
using namespace PacBio::Mongo;
using namespace PacBio::Sensor;
using namespace PacBio::Application;
using namespace PacBio::Mongo::Data;
using namespace PacBio::TraceFile;
using namespace PacBio::DataSource;


TEST(TestTraceSaver, TestA)
{
    const size_t startZmw = 0;
    const size_t startFrame = 0;
    const size_t numZmws = 128;
    const size_t laneWidth = 64;
    const size_t numFrames = 512;
    PacBio::Dev::TemporaryDirectory tmpDir;
    const std::string traceFile = tmpDir.DirName()+"/testA.trc.h5";
    {
        auto writer = std::make_unique<TraceFileWriter>(traceFile, numZmws, numFrames);

        ChipLayoutSimple chipLayout(0, 0, 64, 64, "testlayout");
        auto roi = std::make_unique<SequelRectangularROI>(chipLayout.GetSensorROI());
        std::vector<uint32_t> blockIndices;
        blockIndices.push_back(0);
        TraceSaverBody traceSaver(std::move(writer), std::move(roi), blockIndices);

        BatchDimensions dims;
        dims.lanesPerBatch = 1;
        dims.framesPerBatch = 128;
        dims.laneWidth = laneWidth;

        PacketLayout packetLayout(PacketLayout::LayoutType::BLOCK_LAYOUT_DENSE,
                                  PacketLayout::EncodingFormat::INT16,
                                  std::array<size_t,3>{1, dims.framesPerBatch, dims.laneWidth});

        PacBio::Memory::MallocAllocator allocator;


        auto FillTraceBatch = [&](DataSource::SensorPacket& packet, std::function<int16_t(void)> pattern)
        {
            // fill in the tracebatch with sawtooth test patterns.
            // SPECIAL NOTE. The trace*BATCH* (output from the DataSource)
            // has opposite transpose from trace*BLOCK* (input to the
            // tracefile API). This test checks that
            // the transpose is done correctly.
            for(uint32_t iblock=0; iblock < packet.Layout().NumBlocks(); iblock++)
            {
                int16_t* ptr = reinterpret_cast<int16_t*>(packet.BlockData(iblock).Data());
                for (uint32_t zmw = 0; zmw < dims.laneWidth; zmw++)
                {
                    for (uint32_t frame = 0; frame < dims.framesPerBatch; frame++)
                    {
                        uint32_t j = zmw + frame * dims.laneWidth;
                        ASSERT_LT(j, packet.BlockData(iblock).Count() / sizeof(*ptr));
                        ptr[j] = pattern();
                    }
                }
            }
        };

        {
            // first SensorPacket
            const BatchMetadata meta(0 /* poolid*/,
                                     0 /* firstFrame */,
                                     dims.framesPerBatch - 1 /* lastFrame */,
                                     0 /* firstZmw */
            );

            DataSource::SensorPacket packet(packetLayout,
                                            startZmw,
                                            startFrame,
                                            allocator
            );
            ASSERT_EQ(dims.framesPerBatch * laneWidth * sizeof(int16_t), packet.BytesInBlock());
            ASSERT_EQ(dims.framesPerBatch * laneWidth * sizeof(int16_t), packet.BlockData(0).Count());
            ASSERT_EQ(dims.laneWidth, packet.Layout().BlockWidth());

            int16_t pattern = 0;
            FillTraceBatch(packet, [&pattern](){ return pattern++; });

            Mongo::Data::TraceBatch<int16_t> traceBatch(std::move(packet), meta, dims,
                                                        Cuda::Memory::SyncDirection::Symmetric,
                                                        SOURCE_MARKER());

            traceSaver.Process(traceBatch);
        }
        {
            //second SensorPacket
            const BatchMetadata meta(0 /* poolid*/,
                                     dims.framesPerBatch /* firstFrame */,
                                     dims.framesPerBatch + dims.framesPerBatch - 1 /* lastFrame */,
                                     0 /* firstZmw */
            );

            DataSource::SensorPacket packet(packetLayout,
                                            startZmw,
                                            startFrame,
                                            allocator
            );
            int16_t pattern = 0;
            FillTraceBatch(packet, [&pattern](){ return pattern--; });
            Mongo::Data::TraceBatch<int16_t> traceBatch(std::move(packet), meta, dims,
                                                        Cuda::Memory::SyncDirection::Symmetric,
                                                        SOURCE_MARKER());

            traceSaver.Process(traceBatch);
        }
    }
    {
        TraceFileReader reader(traceFile);

        boost::multi_array<int16_t,1> zmwTrace(boost::extents[numFrames]); // read entire ZMW
        ASSERT_EQ(numFrames,zmwTrace.num_elements());
        reader.Traces().ReadZmw(zmwTrace, 0);
        EXPECT_EQ(0,     zmwTrace[0]);
        EXPECT_EQ(1,     zmwTrace[1]);
        EXPECT_EQ(127,   zmwTrace[127]);
        EXPECT_EQ(0,     zmwTrace[128]);
        EXPECT_EQ(-1,    zmwTrace[129]);
        EXPECT_EQ(-127,  zmwTrace[255]);

        reader.Traces().ReadZmw(zmwTrace, 1);
        EXPECT_EQ(128,   zmwTrace[0]);

        reader.Traces().ReadZmw(zmwTrace, 63);
        EXPECT_EQ(-8191, zmwTrace[255]);

        const auto holexy = reader.Traces().HoleXY();
        ASSERT_EQ(holexy.shape()[0], numZmws);
        ASSERT_EQ(holexy.shape()[1], 2);
        EXPECT_EQ(holexy[0][0], 0);
        EXPECT_EQ(holexy[0][1], 0);
        EXPECT_EQ(holexy[1][0], 0);
        EXPECT_EQ(holexy[1][1], 1);
        EXPECT_EQ(holexy[63][0], 0);
        EXPECT_EQ(holexy[63][1], 63);
        EXPECT_EQ(holexy[numZmws-1][0], 1);
        EXPECT_EQ(holexy[numZmws-1][1], 63);
    }
    tmpDir.Keep();
}
