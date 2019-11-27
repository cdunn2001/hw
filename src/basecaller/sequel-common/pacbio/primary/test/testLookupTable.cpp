#include <gtest/gtest.h>

#include <iostream>
#include <string.h>
#include "json/json.h"
#include "json/reader.h"
#include <pacbio/dev/TemporaryDirectory.h>

#include <pacbio/primary/FileHeaderBuilder.h>
#include <pacbio/primary/BazWriter.h>
#include <pacbio/primary/FileHeader.h>
#include <pacbio/primary/FileHeaderBuilder.h>
#include <pacbio/primary/PacketFieldName.h>
#include <pacbio/primary/Sanity.h>
#include <pacbio/primary/SmartMemory.h>

#include "ArminsFakeMovie.h"

using namespace PacBio::Primary;
using Mb = SequelMetricBlock;

TEST(lookupTable, NUMBER_ByHand)
{
    FileHeaderBuilder fhb= ArminsFakeMovie();

    const std::vector<uint32_t> numberStarts = {0x01000100, 0x02000100, 0x03000100};
    for (size_t i = 0; i < numberStarts.size(); ++i)
    {
        for (uint32_t j = 0; j < 5; ++j)
        {
            fhb.AddZmwNumber(numberStarts[i] + j);
        }
    }

    std::vector<std::pair<uint32_t, uint32_t>> expected;
    for (const auto x : numberStarts)
        expected.emplace_back(x, 5);

    EXPECT_EQ(expected, FileHeaderBuilder::RunLengthEncLUT(fhb.ZmwNumbers()));
}


static FileHeader Read(const std::string& filename)
{
    auto file = SmartMemory::OpenFile(filename.c_str(), "rb");

    // Seek for SANITY block and last position of the file header
    uint64_t headerSize = Sanity::FindAndVerify(file);
    // Wrap header into smrt pointer
    std::vector<char> header(headerSize + 1);

    // Set position indicator to beginning
    std::rewind(file.get());

    // Read file header
    size_t result = std::fread(header.data(), 1, headerSize, file.get());
    if (result != headerSize)
        throw std::runtime_error("Cannot read file header!");

    FileHeader fh(header.data(), headerSize);

    return fh;
}

TEST(lookupTable, NUMBER_FH)
{
    const std::string FILENAME_NUMBER1 = "lut_test_number.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME_NUMBER = tmpdir.DirName() + "/" + FILENAME_NUMBER1;

    FileHeaderBuilder fhb = ArminsFakeMovie();

    const std::vector<uint32_t> numberStarts = {0x01000100, 0x02000100, 0x03000100};
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < numberStarts.size(); ++i)
    {
        for (uint32_t j = 0; j < 5; ++j)
        {
            fhb.AddZmwNumber(numberStarts[i] + j);
            expected.emplace_back(numberStarts[i] + j);
        }
    }

    {
        BazWriter<Mb> writer(FILENAME_NUMBER, fhb, PacBio::Primary::BazIOConfig{}, 100000);
        writer.WaitForTermination();
    }
    const auto fileHeader = Read(FILENAME_NUMBER);
    EXPECT_EQ(expected, fileHeader.ZmwNumbers());
}

TEST(lookupTable, MASK_FH)
{
    const std::string FILENAME_MASK1 = "lut_test_mask.baz";
    static PacBio::Dev::TemporaryDirectory tmpdir;
    std::string FILENAME_MASK = tmpdir.DirName() + "/" + FILENAME_MASK1;

    FileHeaderBuilder fhb= ArminsFakeMovie();

    const std::vector<uint32_t> maskStarts = {0x01000100, 0x02000100, 0x03000100};
    std::vector<uint32_t> expected;
    for (size_t i = 0; i < maskStarts.size(); ++i)
        for (uint32_t j = 0; j < 5; ++j)
            expected.emplace_back(maskStarts[i] + j);

    fhb.AddZmwNumber(500);
    fhb.AddZmwNumber(501);
    fhb.AddZmwNumber(502);
    fhb.ZmwNumberRejects(expected);
    {
        BazWriter<Mb> writer(FILENAME_MASK, fhb, PacBio::Primary::BazIOConfig{}, 100000);
        writer.WaitForTermination();
    }
    const auto fileHeader = Read(FILENAME_MASK);
    EXPECT_EQ(expected, fileHeader.ZmwNumberRejects());

    for (const auto x : expected)
        EXPECT_TRUE(fileHeader.IsZmwNumberRejected(x));

    EXPECT_FALSE(fileHeader.IsZmwNumberRejected(20));
    EXPECT_FALSE(fileHeader.IsZmwNumberRejected(89));
}

