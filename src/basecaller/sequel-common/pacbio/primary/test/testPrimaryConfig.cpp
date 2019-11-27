//
// Created by mlakata on 2/21/17.
//

#include <gtest/gtest.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/primary/PrimaryConfig.h>

using namespace PacBio::Primary;
using namespace PacBio::IPC;

TEST(PrimaryConfig,Test1)
{
    PrimaryConfig pc(0.0);
    pc.chipClass = ChipClass::Sequel;
    pc.platform = Platform::Sequel1PAC1;
    std::string x;
    ASSERT_TRUE(pc.Validate(x));

    // this is the default
    ASSERT_EQ(    16, pc.cache.chunksPerSuperchunk);

    ASSERT_EQ(    16, pc.cache.chunksPerSuperchunk);
    ASSERT_EQ(    16, pc.cache.tilesPerTranche);
    ASSERT_EQ(  8192, pc.cache.framesPerTranche);
    ASSERT_EQ(  8192, pc.cache.framesPerSuperchunk);
    ASSERT_EQ(     8, pc.cache.blocksPerTranche);

    // now change it
    pc.chunksPerSuperchunk = 32;

    ASSERT_EQ(  16384, pc.cache.framesPerTranche);
    ASSERT_EQ(     32, pc.cache.tilesPerTranche);
    ASSERT_EQ(  16384, pc.cache.framesPerTranche);
    ASSERT_EQ(  16384, pc.cache.framesPerSuperchunk);
    ASSERT_EQ(     16, pc.cache.blocksPerTranche);

    // now change it
    pc.Load("chunksPerSuperchunk=8");

    ASSERT_EQ(     8, pc.cache.chunksPerSuperchunk);
    ASSERT_EQ(     8, pc.cache.tilesPerTranche);
    ASSERT_EQ(  4096, pc.cache.framesPerTranche);
    ASSERT_EQ(  4096, pc.cache.framesPerSuperchunk);
    ASSERT_EQ(     4, pc.cache.blocksPerTranche);

    // now change it
    pc.Load("{\"chunksPerSuperchunk\":4}");

    ASSERT_EQ(     4, pc.cache.chunksPerSuperchunk);
    ASSERT_EQ(     4, pc.cache.tilesPerTranche);
    ASSERT_EQ(  2048, pc.cache.framesPerTranche);
    ASSERT_EQ(  2048, pc.cache.framesPerSuperchunk);
    ASSERT_EQ(     2, pc.cache.blocksPerTranche);
}

TEST(PrimaryConfig,DefaultPoolType)
{
    PrimaryConfig pc(0.0);
    std::string x;

    pc.platform = Platform::Sequel1PAC1;
    pc.chipClass = ChipClass::Sequel;
    ASSERT_TRUE(pc.Validate(x)) << x;
    EXPECT_EQ(PoolType::SCIF, pc.poolConfigAcq.GetBestPoolType()) << pc.Json();

    pc.platform = Platform::Sequel1PAC2;
    pc.chipClass = ChipClass::Sequel;
    ASSERT_TRUE(pc.Validate(x)) << x;
    EXPECT_EQ(PoolType::SHM_XSI, pc.poolConfigAcq.GetBestPoolType()) << pc.Json();

    pc.platform = Platform::Spider;
    pc.chipClass = ChipClass::Spider;
    ASSERT_TRUE(pc.Validate(x)) << x;
    EXPECT_EQ(PoolType::SHM_XSI, pc.poolConfigAcq.GetBestPoolType()) << pc.Json();

    pc.platform = Platform::Benchy;
    pc.chipClass = ChipClass::Spider;
    ASSERT_TRUE(pc.Validate(x)) << x;
    EXPECT_EQ(PoolType::SHM_XSI, pc.poolConfigAcq.GetBestPoolType()) << pc.Json();
}
