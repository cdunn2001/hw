//
// Created by mlakata on 1/5/21.
//

#include <gtest/gtest.h>

#include <pacbio/dev/gtest-extras.h>
#include <pacbio/datasource/MallocAllocator.h>

#include <dataTypes/LabelsBatch.h>

using namespace PacBio::Mongo::Data;
using namespace PacBio::DataSource;

TEST(LabelsBatch,Constructor)
{
    // this is a trivial test to just test that the constructor runs without throwing

    BatchMetadata meta(1,2,3,4);
    BatchDimensions dims;
    dims.lanesPerBatch = 10;
    dims.framesPerBatch = 1024;
    TraceBatch<LabelsBatch::ElementType> trace(meta,
                                               dims,
                                               PacBio::Cuda::Memory::SyncDirection::Symmetric,
                                               SOURCE_MARKER());
    LabelsBatch labelsBatch(meta,
                            dims,
                            std::move(trace),
                            512,
                            PacBio::Cuda::Memory::SyncDirection::Symmetric,
                            SOURCE_MARKER());

    EXPECT_EQ(64,labelsBatch.TraceData().LaneWidth());
    EXPECT_EQ(64,labelsBatch.LatentTrace().LaneWidth());
}

TEST(LabelsBatch,Factory)
{
    // trivial test that the factory runs without throwing
    LabelsBatchFactory factory(32,PacBio::Cuda::Memory::SyncDirection::Symmetric);
    BatchMetadata meta(1,2,3,4);
    BatchDimensions dims;
    dims.lanesPerBatch = 10;
    dims.framesPerBatch = 1024;
    TraceBatch<LabelsBatch::ElementType> trace(meta,
                                               dims,
                                               PacBio::Cuda::Memory::SyncDirection::Symmetric,
                                               SOURCE_MARKER());

    auto lb = factory.NewBatch(std::move(trace));
    EXPECT_EQ(64,lb.first.TraceData().LaneWidth());

    auto flm = factory.NewMetrics(dims);
}
