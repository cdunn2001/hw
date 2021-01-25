//
// Created by mlakata on 12/3/20.
//

#include <gtest/gtest.h>

#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/dev/gtest-extras.h>

#include <appModules/Basecaller.h>
#include <appModules/TraceSaver.h>
#include <pacbio/ipc/JSON.h>
#include <pacbio/datasource/MallocAllocator.h>
#include <pacbio/datasource/PacketLayout.h>
#include <pacbio/sensor/RectangularROI.h>
#include <pacbio/sensor/SparseROI.h>
#include <pacbio/tracefile/TraceFile.h>
#include <vector>

using std::vector;

using namespace PacBio;
using namespace PacBio::Mongo;
using namespace PacBio::Sensor;
using namespace PacBio::Application;
using namespace PacBio::Mongo::Data;
// using namespace PacBio::TraceFile;
using namespace PacBio::DataSource;


TEST(TestTraceSaver, TestA)
{
    // The plan of this test is to create a fake sensor of 256 ZMWs (2 rows, 2 lanes = 128 columns), have a trace ROI
    // that only selects the first two columns, and simulate running the smrt-basecaller until completion. Each
    // TraceBatch spans the entire chip and 128 frames. We want to simulate a movie of 1024 frames, so there will need
    // to be several TraceBatches.
    //
    const size_t laneWidth = 64;
    const size_t numFrames = 1024;
    SensorSize sensorROI(0, 0, 2, laneWidth * 2, 1, 1);
    const uint64_t numSensorZmws = sensorROI.NumUnitCells();
    const uint64_t numSelectedZmws = 128;

    const uint32_t numCols = sensorROI.PhysicalCols();
    auto alpha = [](uint32_t row, uint32_t col, uint64_t frame) {
        const int16_t value = (col + row * 10 + frame * 100) % 2048;
        return value;
    };
    auto alphaPrime = [&numCols, &alpha](uint64_t zmwOffset, uint64_t frame) {
        const uint32_t col = zmwOffset % numCols;
        const uint32_t row = zmwOffset / numCols;
        return alpha(row, col, frame);
    };

    PacBio::Dev::TemporaryDirectory tmpDir;
    const std::string traceFile = tmpDir.DirName() + "/testA.trc.h5";
    {
        std::unique_ptr<GenericROI> roi = std::make_unique<RectangularROI>(0, 0, 2, laneWidth, sensorROI);
        ASSERT_EQ(roi->CountZMWs(), numSelectedZmws);

        auto writer = std::make_unique<PacBio::Application::TraceFileWriter>(traceFile, numSelectedZmws, numFrames);

        std::vector<DataSourceBase::LaneIndex> lanes;
        lanes.push_back(0);  // starting at (0,0)
        lanes.push_back(2);  // starting at (1,0)

        std::vector<DataSourceBase::UnitCellFeature> roiFeatures(numSelectedZmws);
        size_t k=0;
        for(const DataSourceBase::LaneIndex lane : lanes)
        {
            for(uint32_t j =0;j<laneWidth; j++)
            {
                roiFeatures[k].flags = 0;
                roiFeatures[k].x = (lane ==0) ? 0 : 1;
                roiFeatures[k].y = (lane ==0) ? j : j;
                k++;
            }
        }
        // which skip (0,64) and (1,64)
        DataSourceBase::LaneSelector laneSelector(lanes);
        TraceSaverBody traceSaver(std::move(writer), roiFeatures, std::move(laneSelector));

        BatchDimensions dims;
        dims.lanesPerBatch = 1;
        dims.framesPerBatch = 128;
        dims.laneWidth = laneWidth;

        PacketLayout packetLayout(PacketLayout::LayoutType::BLOCK_LAYOUT_DENSE,
                                  PacketLayout::EncodingFormat::INT16,
                                  std::array<size_t, 3> {1, dims.framesPerBatch, dims.laneWidth});

        PacBio::Memory::MallocAllocator allocator;


        // fill each packet with the "alpha" test pattern, which is based on row and column
        auto FillTraceBatch = [&](uint64_t zmwOffset,
                                  uint64_t frameOffset,
                                  DataSource::SensorPacket& packet,
                                  std::function<int16_t(uint64_t zmw, uint64_t frame)> pattern) {
            for (uint32_t iblock = 0; iblock < packet.Layout().NumBlocks(); iblock++)
            {
                int16_t* ptr = reinterpret_cast<int16_t*>(packet.BlockData(iblock).Data());
                for (uint32_t zmw = 0; zmw < dims.laneWidth; zmw++)
                {
                    for (uint32_t frame = 0; frame < dims.framesPerBatch; frame++)
                    {
                        uint32_t j = zmw + frame * dims.laneWidth;
                        ASSERT_LT(j, packet.BlockData(iblock).Count() / sizeof(*ptr));
                        ptr[j] = pattern(zmw + zmwOffset, frame + frameOffset);
                    }
                }
            }
        };

        // create all the packets, convert each one to a TraceBatch and process it.
        for (uint64_t iframe = 0; iframe < numFrames; iframe += dims.framesPerBatch)
        {
            for (uint64_t izmw = 0; izmw < numSensorZmws; izmw += dims.laneWidth)
            {
                const BatchMetadata meta(0 /* poolid*/,  // don't care
                                         iframe /* firstFrame */,
                                         iframe + dims.framesPerBatch - 1 /* lastFrame */,
                                         izmw /* firstZmw */
                );

                DataSource::SensorPacket packet(packetLayout, izmw, iframe, allocator);
                ASSERT_EQ(dims.framesPerBatch * laneWidth * sizeof(int16_t), packet.BytesInBlock());
                ASSERT_EQ(dims.framesPerBatch * laneWidth * sizeof(int16_t), packet.BlockData(0).Count());
                ASSERT_EQ(dims.laneWidth, packet.Layout().BlockWidth());

                FillTraceBatch(izmw, iframe, packet, alphaPrime);

#if 0
                auto dataView = packet.BlockData(0);
                std::stringstream ss;
                ss << "packet for zmw:" << izmw << " frame:" << iframe << "\n";
                const int16_t* pixel = reinterpret_cast<int16_t*>(dataView.Data());
                for (uint64_t iframe2 = 0; iframe2 < dims.framesPerBatch; iframe2++)
                {
                    ss << "[" << iframe2 << "]";
                    for (uint64_t izmw2 = 0; izmw2 < dims.laneWidth; izmw2++)
                    {
                        ss << *pixel++ << " ";
                    }
                    ss << std::endl;
                }
                ss << "----" << std::endl;
                TEST_COUT << ss.str();
#endif

                Mongo::Data::TraceBatch<int16_t> traceBatch(
                    std::move(packet), meta, dims, Cuda::Memory::SyncDirection::Symmetric, SOURCE_MARKER());

                traceSaver.Process(traceBatch);  // blah
            }
        }
    }
    {
        PacBio::TraceFile::TraceFile reader(traceFile);

        boost::multi_array<int16_t, 1> zmwTrace(boost::extents[numFrames]);  // read entire ZMW
        ASSERT_EQ(numFrames, zmwTrace.num_elements());

        const auto holexy = reader.Traces().HoleXY();
        ASSERT_EQ(holexy.shape()[0], numSelectedZmws);
        ASSERT_EQ(holexy.shape()[1], 2);

        int failures = 0;
        for (uint64_t izmw = 0; izmw < numSelectedZmws; izmw++)
        {
            const uint32_t row = holexy[izmw][0];
            const uint32_t col = holexy[izmw][1];

            const uint32_t expectedRow = (izmw < 64) ? 0 : 1;
            const uint32_t expectedCol = (izmw % 64);
            ASSERT_EQ(row, expectedRow);
            ASSERT_EQ(col, expectedCol);

            reader.Traces().ReadZmw(zmwTrace, izmw);
            for (uint64_t iframe = 0; iframe < numFrames; iframe++)
            {
                int16_t expectedPixel = alpha(row, col, iframe);
                EXPECT_EQ(expectedPixel, zmwTrace[iframe])
                    << "zmw:" << izmw << " frame:" << iframe << " row:" << row << " col:" << col;
                if (expectedPixel != zmwTrace[iframe])
                    failures++;
                ASSERT_LT(failures, 10) << " too many failures";
            }
        }
    }
    tmpDir.Keep();
}

TEST(Sanity,UpperBound)
{
    std::vector<uint64_t> lanes;

    lanes.push_back(0);
    lanes.push_back(2);
    auto begin = std::lower_bound(lanes.begin(),lanes.end(), 0);
    auto end   = std::upper_bound(begin,lanes.end(), 0);
    EXPECT_EQ(begin , lanes.begin());
    EXPECT_EQ(end, lanes.begin() + 1);
}

TEST(Sanity,ROI)
{
    std::vector<DataSourceBase::LaneIndex> dummy(2);
    dummy.push_back(0);
    dummy.push_back(4); // ??
    DataSourceBase::LaneSelector blocks(dummy);
    EXPECT_EQ(blocks.size(), dummy.size());
    const size_t numZmws = blocks.size() * laneSize;

    std::vector<DataSourceBase::UnitCellFeature> roiFeatures(numZmws);
    size_t k=0;
    for(const DataSourceBase::LaneIndex lane : blocks)
    {
        for(uint32_t j =0;j<laneSize;j++)
        {
            roiFeatures[k].flags = 0;;
            roiFeatures[k].x = (lane ==0) ? 0 : 2;
            roiFeatures[k].y = (lane ==0) ? j : j;
            k++;
        }
    }

    PacBio::Dev::TemporaryDirectory tmpDir;
    const std::string traceFile = tmpDir.DirName() + "/testB.trc.h5";
    PBLOG_INFO << "Opening TraceSaver with output file " << traceFile << ", " << numZmws << " ZMWS.";
    const uint64_t frames=1024;
    auto outputTrcFile = std::make_unique<TraceFileWriter>(traceFile,
                                                        numZmws,
                                                        frames);

    {
        TraceSaverBody body(std::move(outputTrcFile), roiFeatures, std::move(blocks));
    }
    {
        PacBio::TraceFile::TraceFile reader(traceFile);
        auto holexy = reader.Traces().HoleXY();
        for(uint32_t i=0;i<numZmws;i++)
        {
            EXPECT_EQ(holexy[i][0],i<64 ? 0 : 2);
            EXPECT_EQ(holexy[i][1],i%64);
        }
    }
}
