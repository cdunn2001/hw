// Copyright (c) 2020, Pacific Biosciences of California, Inc.
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

        auto writer = std::make_unique<PacBio::TraceFile::TraceFile>(traceFile, TraceFile::TraceDataType::INT16, numSelectedZmws, numFrames);

        std::vector<DataSourceBase::LaneIndex> lanes;
        lanes.push_back(0);  // starting at (0,0)
        lanes.push_back(2);  // starting at (1,0)

        std::vector<DataSourceBase::UnitCellProperties> roiFeatures(numSelectedZmws);
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

        // fill each batch with the "alpha" test pattern, which is based on row and column
        auto FillTraceBatch = [&](uint64_t zmwOffset,
                                  uint64_t frameOffset,
                                  TraceBatch<int16_t>& batch,
                                  std::function<int16_t(uint64_t zmw, uint64_t frame)> pattern) {
            for (uint32_t iblock = 0; iblock < batch.LanesPerBatch(); iblock++)
            {
                auto blockView = batch.GetBlockView(iblock);
                for (uint32_t zmw = 0; zmw < dims.laneWidth; zmw++)
                {
                    for (uint32_t frame = 0; frame < dims.framesPerBatch; frame++)
                    {
                        uint32_t j = zmw + frame * dims.laneWidth;
                        ASSERT_LT(j , blockView.Size());
                        blockView[j] = pattern(zmw + zmwOffset, frame + frameOffset);
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

                Mongo::Data::TraceBatch<int16_t> traceBatch(
                    meta, dims,
                    Cuda::Memory::SyncDirection::Symmetric,
                    SOURCE_MARKER());

                FillTraceBatch(izmw, iframe, traceBatch, alphaPrime);

#if 0
                auto dataView = traceBatch.GetBlockView(0);
                std::stringstream ss;
                ss << "batch for zmw:" << izmw << " frame:" << iframe << "\n";
                const int16_t* pixel = dataView.Data();
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

                traceSaver.Process(std::move(traceBatch));  // blah
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
    std::vector<DataSourceBase::LaneIndex> dummy;
    dummy.push_back(0);
    dummy.push_back(4); // ??
    DataSourceBase::LaneSelector blocks(dummy);
    EXPECT_EQ(blocks.size(), dummy.size());
    const size_t numZmws = blocks.size() * laneSize;

    std::vector<DataSourceBase::UnitCellProperties> roiFeatures(numZmws);
    size_t k=0;
    for(const DataSourceBase::LaneIndex lane : blocks)
    {
        for(uint32_t j =0;j<laneSize;j++)
        {
            roiFeatures[k].flags = 0;;
            roiFeatures[k].x = (lane == 0) ? 0 : 2;
            roiFeatures[k].y = (lane == 0) ? j : j;
            k++;
        }
    }

    PacBio::Dev::TemporaryDirectory tmpDir;
    const std::string traceFileName = tmpDir.DirName() + "/testB.trc.h5";
    const uint64_t frames=1024;
    PBLOG_INFO << "Opening TraceSaver with output file " << traceFileName << ", " << numZmws << " ZMWS.";
    {
        auto outputTrcFile = std::make_unique<PacBio::TraceFile::TraceFile>(traceFileName, TraceFile::TraceDataType::INT16, numZmws, frames);
        TraceSaverBody body(std::move(outputTrcFile), roiFeatures, std::move(blocks));
    }
    {
        PacBio::TraceFile::TraceFile reader(traceFileName);
        EXPECT_EQ(frames, reader.Traces().NumFrames());
        auto holexy = reader.Traces().HoleXY();
        for(uint32_t i=0;i<numZmws;i++)
        {
            EXPECT_EQ(holexy[i][0],i<64 ? 0 : 2);
            EXPECT_EQ(holexy[i][1],i%64);
        }
    }
}
