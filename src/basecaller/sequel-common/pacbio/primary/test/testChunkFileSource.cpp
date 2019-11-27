#include <string>
#include <vector>
#include <numeric>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <pacbio/primary/Chunking.h>
#include <pacbio/primary/ChunkFileSource.h>
#include <pacbio/primary/Tranche.h>
#include <pacbio-primary-test-config.h>  // Defines sequelBaseDir.

#include "testTraceFilePath.h"

using std::string;
using std::vector;
using boost::numeric_cast;
using boost::filesystem::path;
using namespace ::PacBio::Primary;

// Link tests into TestCrystal executable
int LinkTests_ChunkFileSourece() { return EXIT_SUCCESS; }

TEST(TestChunkFileSource, ReadChunkFile)
{
    path sequelDir = PacBio::Primary::CommonTest::sequelBaseDir;
    auto chunkFilePath = sequelDir / "common" / "pacbio" / "primary" / "test";
#ifdef PB_MIC_COPROCESSOR
    chunkFilePath = "/home/pbi/test_64[512]x2k[512].int16";
#else
    chunkFilePath /= "test_64[512]x2k[512].int16";

    if (!boost::filesystem::exists(chunkFilePath))
    {
        auto fp = TraceFilePath("/dept/primary/unitTestInput/common/pacbio/primary/test/test_64[512]x2k[512].int16");
        chunkFilePath = boost::filesystem::path(fp);
    }
#endif

    // Values for this test file (created in MATLAB)
    const size_t TestFileNumLanes = 2;
    const size_t TestFileNumChunks = 4;
    const size_t TestFileChannelsPerLane = 32;
    const size_t TestFileSampleSize = 64; // bytes

    const size_t Lane0Checksum = 1473654910;
    const size_t Lane1Checksum = 1473825094;
    const size_t TestFileNumFrames = 2048;

    Chunking dims;
    {
        // Read the initial header to get the file dimensions
        std::ifstream file(chunkFilePath.c_str(), std::ios::in | std::ios::binary);
        assert(file.good());
        file >> dims;

        // Files for this test should be the standard simd_pixel lane size.
        ASSERT_EQ(dims.sizeofSample, TestFileChannelsPerLane * sizeof(int16_t));
        ASSERT_EQ(dims.sizeofSample, TestFileSampleSize);

        // Lanes were initially called "traces", short for SIMD-trace.
        EXPECT_EQ(TestFileNumLanes, dims.laneNum);
        EXPECT_EQ(TestFileNumChunks, dims.chunkNum);
    }
    
    // Create a source to read from
    ChunkFileSource src(chunkFilePath.string());

    // Container for doing block reads
    size_t nFrames = dims.sampleNum;
    vector<Tranche::Pixelz::SimdPixel> buf(nFrames);

    // Checksum variables
    size_t frames_read = 0;
    size_t total_frames = 0;
    vector<size_t> laneChecksum(dims.laneNum, 0);
    bool terminus;

    do // Read the next block
    {
        size_t laneIndex;
        size_t chunkIndex;
        bool chunkCompleted;
        
        // Get size and indexing info
        nFrames = src.NextBlock(laneIndex, chunkIndex, chunkCompleted, terminus);
        buf.resize(nFrames);

        // Read into the buffer
        frames_read = src.ReadBlock(buf.data(), nFrames);
        assert(frames_read == nFrames);

        if (laneIndex == 0)
            total_frames += frames_read;

        size_t blockChecksum = 0;
        for (size_t i = 0; i<frames_read; ++i)
        {
            auto v = buf[i].i16;
            for (size_t j = 0; j < TestFileChannelsPerLane; ++j)
                blockChecksum += v[j];
        }

        laneChecksum[laneIndex] += blockChecksum;

    } while (frames_read > 0);
    
    EXPECT_TRUE(terminus);
    EXPECT_EQ(laneChecksum[0], Lane0Checksum);
    EXPECT_EQ(laneChecksum[1], Lane1Checksum);
    EXPECT_EQ(total_frames, TestFileNumFrames);
}
