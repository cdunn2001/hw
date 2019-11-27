
#include <unistd.h>
#include <fstream>

#include <gtest/gtest.h>
#include <pacbio/dev/gtest-extras.h>
#include <pacbio/dev/AutoTimer.h>

#include <pacbio/primary/ManuallyBufferedFile.h>

using namespace PacBio::Primary;
using namespace PacBio::Dev;

TEST(ManuallyBufferedFile, BufferedSmallWrites)
{
    TempFileManager tfm;
    const auto& tmpName = tfm.GenerateTempFileName("");

    // Don't really care what's in here, we just want just under 1MB of data to write
    std::vector<uint8_t> trashData(1000000);

    // file with 4Mb buffers, an allowing a 100Gb/s writes to keep us unbound there
    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 4, 100000));

    struct stat fileStat;
    // Make sure file even exists now
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));

    // Should be able to do 4 1k writes before tripping a flush
    for (size_t i = 0; i < 4; ++i)
    {
        file->Fwrite(trashData.data(), trashData.size());
        ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
        ASSERT_EQ(0, fileStat.st_size);
    }

    // This write wont fit in the buffer, so the previous writes need to get
    // written first
    file->Fwrite(trashData.data(), trashData.size());
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(4000000, fileStat.st_size);

    // Next write shouldn't trigger anything
    file->Fwrite(trashData.data(), trashData.size());
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(4000000, fileStat.st_size);

    // But closing the file should flush everything out
    file.reset();
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(6000000, fileStat.st_size);
}

TEST(ManuallyBufferedFile, SmallAndLargeWrite)
{
    TempFileManager tfm;
    const auto& tmpName = tfm.GenerateTempFileName("");

    // junk data to write
    std::vector<uint8_t> smallTrashData(1000000);
    std::vector<uint8_t> largeTrashData(10000000);

    // file with 4Mb buffers, an allowing a 100Gb/s writes to keep us unbound there
    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 4, 100000));

    struct stat fileStat;

    // First write small enough to buffer
    file->Fwrite(smallTrashData.data(), smallTrashData.size());
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(0, fileStat.st_size);

    // But the large write will flush everything out
    file->Fwrite(largeTrashData.data(), largeTrashData.size());
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(11000000, fileStat.st_size);
}

TEST(ManuallyBufferedFile, LargeWrite)
{
    TempFileManager tfm;
    const auto& tmpName = tfm.GenerateTempFileName("");

    // junk data to write
    std::vector<uint8_t> largeTrashData(10000000);

    // file with 4Mb buffers, an allowing a 100Gb/s writes to keep us unbound there
    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 4, 100000));

    struct stat fileStat;

    // Does not fit in buffer, must be written directly to file
    file->Fwrite(largeTrashData.data(), largeTrashData.size());
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(10000000, fileStat.st_size);
}

TEST(ManuallyBufferedFile, ThrottledWrite)
{
    TempFileManager tfm("/dev/shm","ThrottledWrite");
    const auto& tmpName = tfm.GenerateTempFileName("");

    // junk data to write. 1 byte over the buffer size of the file object.
    std::vector<uint8_t> largeTrashData((1 << 20) + 1);

    // file with 4Mb buffers, an allowing only 100MB per second to keep us throttled
    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 1, 10));

    struct stat fileStat;
    // Does not fit in buffer, must be written directly to file
    AutoTimer timer;
    file->Fwrite(largeTrashData.data(), largeTrashData.size());
    auto timeTaken = timer.GetElapsedMilliseconds();

    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    // should have taken 100ms. but give it some wiggle room
    ASSERT_LT(80, timeTaken);
    ASSERT_GT(200, timeTaken);
//    TEST_COUT << "timeTaken1:" << timeTaken << std::endl;


    timer.Restart();
    file->Fwrite(largeTrashData.data(), largeTrashData.size());
    file->Fwrite(largeTrashData.data(), largeTrashData.size());
    timeTaken = timer.GetElapsedMilliseconds();

    // should have taken 200ms. but give it some wiggle room
    ASSERT_LT(180, timeTaken);
    ASSERT_GT(300, timeTaken);
//    TEST_COUT << "timeTaken2:" << timeTaken << std::endl;

}

TEST(ManuallyBufferedFile, PatternWrite)
{
    TempFileManager tfm;
    const auto& tmpName = tfm.GenerateTempFileName("");

    // junk data to write
    std::vector<char> data(1024);
    for (size_t i = 0; i < 1024; ++i) data[i] = i;

    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 4, 100000));
    for (size_t i = 0; i < 10; ++i)
    {
        file->Fwrite(data.data(), data.size());
    }
    file.reset();

    struct stat fileStat;
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(10240, fileStat.st_size);

    std::ifstream input(tmpName, std::ios::in | std::ios::binary);
    std::vector<char> checkData(1024);
    for (size_t i = 0; i < 10; ++i)
    {
        input.read(checkData.data(), checkData.size());
        for (size_t j = 0; j < checkData.size(); ++j)
        {
            ASSERT_EQ(checkData[j], data[j]);
        }
    }

}

TEST(ManuallyBufferedFile, PatternWriteSeek)
{
    TempFileManager tfm;
    const auto& tmpName = tfm.GenerateTempFileName("");

    // junk data to write
    std::vector<char> data(1024);
    for (size_t i = 0; i < 1024; ++i) data[i] = i;

    // Write the pattern 10 times, but mark the start of the 10th repeat
    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 4, 100000));
    size_t midpoint;
    for (size_t i = 0; i < 10; ++i)
    {
        file->Fwrite(data.data(), data.size());
        if (i == 4) midpoint = file->Ftell();
    }

    // Jump recorded location, and rewrite that repeat with 0s
    file->FseekAbsolute(midpoint);
    std::vector<char> data0(1024, 0);
    file->Fwrite(data0.data(), data0.size());

    // Jump to the previous pattern, and overwrite it with 1s
    file->FseekRelative(-2048);
    std::vector<char> data1(1024, 1);
    file->Fwrite(data1.data(), data1.size());

    // Now jump ahead and overwrite with 2s
    file->FseekRelative(1024);
    std::vector<char> data2(1024, 2);
    file->Fwrite(data2.data(), data2.size());

    file.reset();

    struct stat fileStat;
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(10240, fileStat.st_size);

    std::ifstream input(tmpName, std::ios::in | std::ios::binary);
    std::vector<char> checkData(1024);
    for (size_t i = 0; i < 10; ++i)
    {
        input.read(checkData.data(), checkData.size());
        for (size_t j = 0; j < checkData.size(); ++j)
        {
            if (i == 4)
                ASSERT_EQ(checkData[i], 1);
            else if (i == 5)
                ASSERT_EQ(checkData[i], 0);
            else if (i == 6)
                ASSERT_EQ(checkData[i], 2);
            else
                ASSERT_EQ(checkData[i], data[i]);
        }
    }
}

TEST(ManuallyBufferedFile, PatternWriteAdvance)
{
    TempFileManager tfm;
    const auto& tmpName = tfm.GenerateTempFileName("");

    // junk data to write
    std::vector<char> data(1024);
    for (size_t i = 0; i < 1024; ++i) data[i] = i;

    std::unique_ptr<ManuallyBufferedFile> file(new ManuallyBufferedFile(tmpName, 4, 100000));
    // fill in a set of zeroes at the start
    file->Advance(1024);

    for (size_t i = 0; i < 10; ++i)
    {
        file->Fwrite(data.data(), data.size());
    }
    file.reset();

    struct stat fileStat;
    ASSERT_EQ(0, stat(tmpName.c_str(), &fileStat));
    ASSERT_EQ(11264, fileStat.st_size);

    std::ifstream input(tmpName, std::ios::in | std::ios::binary);
    std::vector<char> checkData(1024);
    for (size_t i = 0; i < 11; ++i)
    {
        input.read(checkData.data(), checkData.size());
        for (size_t j = 0; j < checkData.size(); ++j)
        {
            if (i == 0)
                ASSERT_EQ(checkData[j], 0);
            else
                ASSERT_EQ(checkData[j], data[j]);
        }
    }

}
