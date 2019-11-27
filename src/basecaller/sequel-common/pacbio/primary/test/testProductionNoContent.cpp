#include <gtest/gtest.h>

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <assert.h>

#include <pacbio/dev/TemporaryDirectory.h>
#include <pacbio/smrtdata/Basecall.h>
#include <pacbio/smrtdata/Pulse.h>

#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/BazReader.h>
#include <pacbio/primary/BazCore.h>
#include <pacbio/primary/SmartMemory.h>
#include <pacbio/primary/Timing.h>
#include <pacbio/primary/PrimaryToBaz.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using Mb = SequelMetricBlock;

static void write(const std::string& filename)
{
    FileHeaderBuilder fhb = ArminsFakeMovie();
    fhb.AddZmwNumber(0);
    BazWriter<Mb> writer(filename, fhb, PacBio::Primary::BazIOConfig{}, 100000);
    EXPECT_TRUE(writer.AddZmwSlice(nullptr, 0, std::vector<Mb>(), 0));
    EXPECT_TRUE(writer.Flush());
    writer.WaitForTermination();
}

TEST(productionNoContent, test)
{
    const std::string FILENAME1 = "prodNoContent_test.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME = tmpdir.DirName() + "/" + FILENAME1;

    write(FILENAME);
    BazReader reader(FILENAME);
    EXPECT_TRUE(reader.HasNext());
    // This is actually the terminal SuperChunkMeta
    auto zmws = reader.NextSlice();
    EXPECT_EQ(zmws.size(), static_cast<size_t>(0));
    EXPECT_FALSE(reader.HasNext());
    EXPECT_FALSE(reader.HasNext());
}

