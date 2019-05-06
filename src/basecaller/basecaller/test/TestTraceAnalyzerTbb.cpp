
#include <vector>
#include <gtest/gtest.h>

#include <ITraceAnalyzer.h>
#include <BasecallerConfig.h>
#include <BatchMetadata.h>
#include <MovieConfig.h>

using std::vector;

namespace PacBio {
namespace Mongo {
namespace Basecaller {

TEST(TestTraceAnalyzerTbb, CheckMetadata)
{
    const unsigned int numPools = 8;
    Data::BasecallerAlgorithmConfig bcConfig;
    Data::MovieConfig movConfig;
    auto traceAnalyzer = ITraceAnalyzer::Create(numPools, bcConfig, movConfig);

    ASSERT_EQ(numPools, traceAnalyzer->NumZmwPools());

    const Data::BatchDimensions dims {64, 16, 4};
    vector<Data::TraceBatch<int16_t>> chunk;
    vector<Data::BatchMetadata> bmdVec;
    // Notice that we skip some pool ids.
    for (unsigned int i = 0; i < numPools; i += 2)
    {
        const Data::BatchMetadata bmd(i, 0, dims.framesPerBatch);
        bmdVec.push_back(bmd);
        chunk.emplace_back(bmd, dims, Cuda::Memory::SyncDirection::Symmetric);
    }

    // The function under test.
    const auto bcBatch = (*traceAnalyzer)(std::move(chunk));

    ASSERT_EQ(bmdVec.size(), bcBatch.size());
    for (unsigned int i = 0; i < bcBatch.size(); ++i)
    {
        EXPECT_EQ(bmdVec[i], bcBatch[i].GetMeta());
    }
}

}}}     // namespace PacBio::Mongo::Basecaller
